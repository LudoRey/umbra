from pathlib import Path
from collections.abc import Sequence
from typing import Any, cast

import dateutil.parser
import numpy as np
import os
import astropy.io.fits
from umbra.common import coords, convert
from umbra.common.terminal import cprint


FITS_EXTENSIONS = frozenset({".fits", ".fit"})


def read_fits(filepath: Path | str, region: coords.Region | None = None, to_float=True, verbose=True, *, checkstate=lambda: None):
    if verbose:
        cprint(f"Opening {filepath}...")
    # Open image/header
    with astropy.io.fits.open(filepath) as hdul:
        hdu = cast(astropy.io.fits.PrimaryHDU, hdul[0])
        hdu.verify('silentfix')
        header = hdu.header
        data = hdu.data
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Found no image data in {filepath}.")
        if region is None:
            img = data
        else:
            if data.ndim == 2:
                img = data[region.top:region.bottom, region.left:region.right]
            else:
                img = data[:, region.top:region.bottom, region.left:region.right]
    if to_float:
        img = convert.to_float(img)
    # If color image : CxHxW -> HxWxC
    if len(img.shape) == 3:
        img = np.moveaxis(img, 0, 2)
    checkstate()
    return img, header

def remove_pedestal(img: np.ndarray, header: astropy.io.fits.Header) -> np.ndarray:
    # Updates header in-place
    pedestal = header.get("PEDESTAL")
    if pedestal is not None and isinstance(pedestal, (int, float)):
        img = img - pedestal / 65535
        img = np.maximum(img, 0)
        header.remove("PEDESTAL")
    return img

def extract_timestamp(header: astropy.io.fits.Header) -> float:
    timestr = header.get("DATE-OBS")
    if timestr is None:
        raise ValueError("FITS header does not contain a DATE-OBS keyword.")
    return dateutil.parser.parse(cast(str, timestr)).timestamp()

def save_as_fits(img: np.ndarray, header: astropy.io.fits.Header | None, filepath: Path | str, convert_to_uint16=False, verbose=True, *, checkstate=lambda: None):
    if verbose:
        cprint(f"Writing {filepath}...")
    if convert_to_uint16:
        img = convert.to_uint16(img)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if len(img.shape) == 3:
        img = np.moveaxis(img, 2, 0)
    hdu = astropy.io.fits.PrimaryHDU(data=img, header=header)
    hdu.writeto(filepath, overwrite=True)
    checkstate()

def read_fits_header(filepath: Path | str, verbose=False) -> astropy.io.fits.Header:
    if verbose:
        cprint(f"Opening {filepath}...")
    with astropy.io.fits.open(filepath) as hdul:
        hdu = cast(astropy.io.fits.PrimaryHDU, hdul[0])
        hdu.verify('silentfix')
        header = hdu.header
    return header

def list_fits_filepaths(dirpath: Path | str) -> list[Path]:
    dirpath = Path(dirpath)
    return [
        p
        for p in dirpath.iterdir()
        if p.is_file() and p.suffix in FITS_EXTENSIONS
    ]

def update_header(header: astropy.io.fits.Header, cards: Sequence[astropy.io.fits.Card], in_place=False):
    header = astropy.io.fits.Header(header, copy = not in_place)
    header.extend(cards, strip=False, update=True)
    return header

def combine_headers(headers: Sequence[astropy.io.fits.Header]):
    header = astropy.io.fits.Header()
    for header in headers:
        header.extend(header, strip=False, update=True)
    return header

def intersect_headers(headers: Sequence[astropy.io.fits.Header]):
    def hash_card(card: astropy.io.fits.Card):
        return hash((card.keyword, card.value, card.comment))
    
    hashes = [set(map(hash_card, header.cards)) for header in headers]
    common = set.intersection(*hashes)

    return astropy.io.fits.Header([card for card in headers[0].cards if hash_card(card) in common])

def get_grouped_filepaths(filepath_to_header: dict[Path, astropy.io.fits.Header], keywords: Sequence[str]) -> dict[tuple[str, ...], list[Path]]:
    """
    Groups FITS filepaths according to the values of specified header keywords.

    Takes a dict of filepath -> header and a list of keywords.

    Example : for keywords = ["EXPTIME", "ISOSPEED"], the returned dict will have the following structure
    {("0.25","100"): ["filename1.fits", "filename2.fits"], ("1","100"): ["filename3.fits"], ("1","200"): ["filename4.fits"]}
    """
    if len(keywords) == 0:
        return {(): list(filepath_to_header.keys())}
    nested_dict = {}
    for filepath, header in filepath_to_header.items():
        # Create/access deepest level of nested dict
        sub_dict = nested_dict
        for keyword in keywords[:-1]:
            group_key = header[keyword]
            if group_key not in sub_dict.keys():
                sub_dict[group_key] = {}
            sub_dict = sub_dict[group_key]
        # Deepest level : sub_dict is a simple dict with filepaths lists as values, and last keyword values as keys (e.g. the ISO value)
        keyword = keywords[-1]
        group_key = header[keyword]
        if group_key in sub_dict.keys():
            sub_dict[group_key].append(filepath)
        else:
            sub_dict[group_key] = [filepath]
    # Sort nested dict
    nested_dict = sort_nested_dict(nested_dict)
    # Collapse dict
    collapsed_dict = collapse_nested_dict(nested_dict)
    # Format keys
    collapsed_dict = format_collapsed_dict_keys(collapsed_dict, keywords)
    return collapsed_dict


def sort_nested_dict(nested_dict: dict):
    for key in nested_dict.keys():
        if isinstance(nested_dict[key], dict):
            nested_dict[key] = sort_nested_dict(nested_dict[key])
    nested_dict = dict(sorted(nested_dict.items()))

    return nested_dict

def collapse_nested_dict(nested_dict: dict):
    # We cant use lists as keys since they are not hashable, so we use tuples instead
    collapsed_dict = {}
    for key in nested_dict.keys():
        if isinstance(nested_dict[key], dict):
            collapsed_subdict = collapse_nested_dict(nested_dict[key])
            for subkey in collapsed_subdict.keys():
                collapsed_dict[(key,)+subkey] = collapsed_subdict[subkey] 
        else: # in that case, nested_dict is a regular dict and only the keys need to be formatted into tuples
            collapsed_dict[(key,)] = nested_dict[key]
    return collapsed_dict

def format_collapsed_dict_keys(collapsed_dict: dict, keywords: Sequence[str]):
    formatted_dict = {}
    for key_tuple, value in collapsed_dict.items():
        formatted_key_tuple = tuple(format_keyword_value(k, keywords[i]) for i, k in enumerate(key_tuple))
        formatted_dict[formatted_key_tuple] = value
    return formatted_dict

def format_keyword(keyword: str) -> str:
    if keyword == "EXPTIME" or keyword == "EXPOSURE":
        return "Exposure"
    elif keyword == "ISOSPEED":
        return "ISO"
    elif keyword == "GAIN":
        return "Gain"
    elif keyword == "DATE-OBS":
        return "Timestamp"
    else:
        return keyword

def format_keyword_value(keyword_value: Any, keyword: str) -> str:
    if keyword == "EXPTIME" or keyword == "EXPOSURE":
        return f"{keyword_value:.5f}"
    elif keyword == "ISOSPEED" or keyword == "GAIN":
        return f"{int(keyword_value)}"
    elif keyword == "DATE-OBS":
        return keyword_value.replace("T", " ")
    else:
        return f"{keyword_value}"
