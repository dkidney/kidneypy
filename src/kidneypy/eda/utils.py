from typing import List

import pandas as pd
from pandas.api.types import (
    is_integer_dtype,
    is_float_dtype,
    is_bool_dtype,
    is_string_dtype,
    is_object_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)

def to_nullable_dtype(s: pd.Series) -> pd.Series:
    """Map a pandas dtype to its nullable equivalent, preserving width."""

    s = pd.Series(s)
    
    input_dtype = str(s.dtype)

    mapping = {
        "int8"   : "Int8",
        "int16"  : "Int16",
        "int32"  : "Int32",
        "int64"  : "Int64",
        "int"    : "Int64",
        "uint8"  : "UInt8",
        "uint16" : "UInt16",
        "uint32" : "UInt32",
        "uint64" : "UInt64",
        "float32": "Float32",
        "float64": "Float64",
        "float"  : "Float64",
        "object" : "string",
        "str"    : "string",
        "bool"   : "boolean",
    }

    # early return if 
    # - dtype is already a nullable dtype
    # - or dtype does not have a nullable equivalent
    if input_dtype in mapping.values() or input_dtype not in mapping.keys():
        return s

    output_dtype = mapping[input_dtype]
    
    return s.astype(output_dtype)


def replace_na(s: pd.Series, values_to_replace: List = None) -> pd.Series:
    """
    Replace specified values with pd.NA, ensuring nullable dtype
    while preserving numeric width.
    """
    
    s = pd.Series(s)

    if isinstance(values_to_replace, list):
        list()

    if not values_to_replace:
        return s

    s = to_nullable_dtype(s)

    if is_datetime64_any_dtype(s.dtype) or is_timedelta64_dtype(s.dtype):
        na_value = pd.NaT
    else:
        na_value = pd.NA

    mask = s.isin(pd.Series(values_to_replace))
    s = s.mask(mask, na_value)

    return s