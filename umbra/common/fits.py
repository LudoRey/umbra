from pathlib import Path
from collections.abc import Sequence
from typing import Any, cast

import dateutil.parser
import astropy.io.fits


Header = astropy.io.fits.Header


def extract_shape(header: Header) -> tuple[int, ...]:
    """Extract the (H, W) or (H, W, C) shape of the image from a FITS header."""
    if "NAXIS1" not in header or "NAXIS2" not in header:
        raise ValueError("FITS header is missing NAXIS1 and/or NAXIS2 keywords.")
    num_channels = header.get("NAXIS3")
    if num_channels is not None:
        return cast(tuple[int, int, int], (header["NAXIS2"], header["NAXIS1"], header["NAXIS3"]))
    return cast(tuple[int, int], (header["NAXIS2"], header["NAXIS1"]))


def extract_timestamp(header: Header) -> float:
    timestr = header.get("DATE-OBS")
    if timestr is None:
        raise ValueError("FITS header does not contain a DATE-OBS keyword.")
    return dateutil.parser.parse(cast(str, timestr)).timestamp()


def extract_bayer_pattern(header: Header) -> str | None:
    """Return the BAYERPAT pattern (uppercased) if present, else None.

    Validation against supported patterns is deferred to :func:`umbra.common.bayer.debayer`.
    """
    value = header.get("BAYERPAT")
    if value is None:
        return None
    return str(value).strip().upper()


def update(header: Header, cards: Sequence[astropy.io.fits.Card], in_place=False) -> Header:
    header = Header(header, copy=not in_place)
    header.extend(cards, strip=False, update=True)
    return header


def combine(headers: Sequence[Header]) -> Header:
    """Merge a sequence of headers into one, later headers overriding earlier ones."""
    result = Header()
    for header in headers:
        result.extend(header, strip=False, update=True)
    return result


def intersect(headers: Sequence[Header]) -> Header:
    def hash_card(card: astropy.io.fits.Card):
        return hash((card.keyword, card.value, card.comment))

    hashes = [set(map(hash_card, header.cards)) for header in headers]
    common = set.intersection(*hashes)

    return Header([card for card in headers[0].cards if hash_card(card) in common])


def group_filepaths(filepath_to_header: dict[Path, Header], keywords: Sequence[str]) -> dict[tuple[str, ...], list[Path]]:
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
    nested_dict = _sort_nested(nested_dict)
    collapsed_dict = _collapse_nested(nested_dict)
    collapsed_dict = _format_collapsed_keys(collapsed_dict, keywords)
    return collapsed_dict


def _sort_nested(nested_dict: dict) -> dict:
    for key in nested_dict.keys():
        if isinstance(nested_dict[key], dict):
            nested_dict[key] = _sort_nested(nested_dict[key])
    nested_dict = dict(sorted(nested_dict.items()))
    return nested_dict


def _collapse_nested(nested_dict: dict) -> dict:
    # We cant use lists as keys since they are not hashable, so we use tuples instead
    collapsed_dict = {}
    for key in nested_dict.keys():
        if isinstance(nested_dict[key], dict):
            collapsed_subdict = _collapse_nested(nested_dict[key])
            for subkey in collapsed_subdict.keys():
                collapsed_dict[(key,)+subkey] = collapsed_subdict[subkey]
        else:  # in that case, nested_dict is a regular dict and only the keys need to be formatted into tuples
            collapsed_dict[(key,)] = nested_dict[key]
    return collapsed_dict


def _format_collapsed_keys(collapsed_dict: dict, keywords: Sequence[str]) -> dict:
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
