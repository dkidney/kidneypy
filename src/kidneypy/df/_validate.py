import numpy as np


def validate_number(x:int|float,
                    name:str='value',
                    is_int:bool=True,
                    lower:int|float=-np.inf,
                    upper:int|float=np.inf,
                    include_lower:bool=True,
                    include_upper:bool=True,
                    finite:bool=True):

    # check type
    try:
        y = int(x) if is_int else float(x)
    except ValueError | OverflowError:
        raise TypeError(f"Expecting {name} to be coercible to {'int' if is_int else 'float'}"
                        f": got {type(x).__name__} instead")

    # check lower bound
    passses_lower = y >= lower if include_lower else y > lower
    if not passses_lower:
        raise ValueError(f"Expecting {name} to be {'>=' if include_lower else '>'} than {lower}")

    # check upper bound
    passes_upper = y <= upper if include_upper else y < upper
    if not passes_upper:
        raise ValueError(f"Expecting {name} to be {'<=' if include_upper else '<'} than {upper}")

    # check finite
    passes_finite = np.isfinite(y) if finite else True
    if not passes_finite:
        raise ValueError(f"Expecting {name} to be finite")

    return y

# float --------------------------------------------------------------------------------------------

def validate_float(x:float, **kwargs):
    return validate_number(x, is_int=False, **kwargs)


def validate_positive_float(x:float, **kwargs):
    return validate_number(x, is_int=False, lower=0.0, include_lower=False, **kwargs)


def validate_non_negative_float(x:float, **kwargs):
    return validate_number(x, is_int=False, lower=0.0, include_lower=True, **kwargs)

# int ----------------------------------------------------------------------------------------------

def validate_int(x:int, **kwargs):
    return validate_number(x, is_int=True, **kwargs)


def validate_positive_int(x:int, **kwargs):
    return validate_number(x, is_int=True, lower=0, include_lower=False, **kwargs)


def validate_non_negative_int(x:int, **kwargs):
    return validate_number(x, is_int=True, lower=0, include_lower=True, **kwargs)

# misc ---------------------------------------------------------------------------------------------

def validate_binary(x:int, **kwargs):
    return validate_int(x, lower=0, upper=1, **kwargs)

def validate_proportion(x:float, **kwargs):
    return validate_float(x, lower=0.0, upper=1.0, **kwargs)
