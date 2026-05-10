from pathlib import Path

import numpy as np
import warnings
import os
import astropy.io.fits
from umbra.common.terminal import cprint

def read_fits(filepath, verbose=True):
    if verbose:
        cprint(f"Opening {filepath}...", color="cyan")
    # Open image/header
    with astropy.io.fits.open(filepath) as hdul:
        header = hdul[0].header
        img = hdul[0].data
    # If color image : CxHxW -> HxWxC
    if len(img.shape) == 3:
        img = np.moveaxis(img, 0, 2)
    return img, header

def read_fits_as_float(filepath, rows_range=None, verbose=True, *, checkstate=lambda: None):
    if verbose:
        cprint(f"Opening {filepath}...", color="cyan")
    # Open image/header
    with astropy.io.fits.open(filepath) as hdul:
        header = hdul[0].header
        if rows_range is None:
            img = hdul[0].data
        else:
            img = hdul[0].data[:,rows_range[0]:rows_range[1]]
    # Type checking and float conversion
    dtype = img.dtype
    img = img.astype(np.float32)
    if np.issubdtype(dtype, np.unsignedinteger) or np.issubdtype(dtype, np.integer): 
        img /= np.iinfo(dtype).max
        if np.issubdtype(dtype, np.integer) and not np.issubdtype(dtype, np.unsignedinteger):
            if img.min() < 0:
                raise ValueError(f"FITS image is in signed integer format and contains negative values.")
            warnings.warn(
                "FITS image is in true signed integer format, which is not officially supported. "
                "Consider converting to unsigned integer (through BZERO trick) or floating point.",
                UserWarning
            )
    elif np.issubdtype(img.dtype, np.floating):
        if img.min() < 0 or img.max() > 1:
            raise ValueError("FITS image is in floating point format but contains values outside the [0,1] range.")
    else:
        raise ValueError(f"Unrecognized FITS image format. Expected integer, unsigned integer, or floating point.")
    # If color image : CxHxW -> HxWxC
    if len(img.shape) == 3:
        img = np.moveaxis(img, 0, 2)
    checkstate()
    return img, header

def remove_pedestal(img, header):
    '''Updates header in-place'''
    if "PEDESTAL" in header:
        img = img - header["PEDESTAL"] / 65535
        img = np.maximum(img, 0)
        del header["PEDESTAL"]
    return img

def save_as_fits(img, header, filepath, convert_to_uint16=False, verbose=True, *, checkstate=lambda: None):
    if verbose:
        cprint(f"Saving as {filepath}...", color="cyan")
    if np.issubdtype(img.dtype, np.uint16):
        pass
    elif np.issubdtype(img.dtype, np.floating):
        if convert_to_uint16:
            img = (np.clip(img, 0, 1)*65535).astype('uint16')
    else:
        raise ValueError(f"Image format must be either 16-bit unsigned integer, or floating point.")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if len(img.shape) == 3:
        img = np.moveaxis(img, 2, 0)
    hdu = astropy.io.fits.PrimaryHDU(data=img, header=header)
    hdu.writeto(filepath, overwrite=True)
    checkstate()

def read_fits_header(filepath, verbose=False, cache=False):
    if verbose:
        cprint(f"Opening {filepath}...", color="cyan")
    with astropy.io.fits.open(filepath, cache=cache) as hdul:
        header = hdul[0].header
    return header

def update_header(header: astropy.io.fits.Header, cards: list[astropy.io.fits.Card], in_place=False):
    header = astropy.io.fits.Header(header, copy = not in_place)
    header.extend(cards, strip=False, update=True)
    return header

def combine_headers(headers: list[astropy.io.fits.Header]):
    header = astropy.io.fits.Header()
    for header in headers:
        header.extend(header, strip=False, update=True)
    return header

def intersect_headers(headers: list[astropy.io.fits.Header]):
    def hash_card(card: astropy.io.fits.Card):
        return hash((card.keyword, card.value, card.comment))
    
    hashes = [set(map(hash_card, header.cards)) for header in headers]
    common = set.intersection(*hashes)

    return astropy.io.fits.Header([card for card in headers[0].cards if hash_card(card) in common])

def get_grouped_filepaths(dirpath: Path | str, keywords: list[str]) -> dict[tuple[str, ...], list[Path]]:
    """
    Groups FITS filepaths in a directory according to the values of specified header keywords.

    Example : for keywords = ["EXPTIME", "ISOSPEED"], the returned dict will have the following structure
    {("0.25","100"): ["filename1.fits", "filename2.fits"], ("1","100"): ["filename3.fits"], ("1","200"): ["filename4.fits"]}
    """
    dirpath = Path(dirpath)
    nested_dict = {}
    for p in dirpath.iterdir():
        if p.is_file() and p.suffix in ('.fits', '.fit'):
            filepath = p
            header = read_fits_header(filepath)
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

def format_collapsed_dict_keys(collapsed_dict: dict, keywords: list[str]):
    formatted_dict = {}
    for key_tuple, value in collapsed_dict.items():
        formatted_key_tuple = tuple(format_keyword_value(k, keywords[i]) for i, k in enumerate(key_tuple))
        formatted_dict[formatted_key_tuple] = value
    return formatted_dict

def format_keyword(keyword):
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

def format_keyword_value(keyword_value, keyword):
    if keyword == "EXPTIME" or keyword == "EXPOSURE":
        return f"{keyword_value:.5f}"
    elif keyword == "ISOSPEED" or keyword == "GAIN":
        return f"{int(keyword_value)}"
    elif keyword == "DATE-OBS":
        return keyword_value.replace("T", " ")
    else:
        return f"{keyword_value}"
