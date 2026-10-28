from kidneypy.df._validate import (
    validate_binary,
    validate_float,
    validate_int,
    validate_non_negative_float,
    validate_non_negative_int,
    validate_positive_float,
    validate_positive_int,
    validate_proportion,
)

import numpy as np
import pytest

# float --------------------------------------------------------------------------------------------

def test_validate_float():

    assert validate_float(1.0) == 1.0
    assert validate_float(1) == 1.0
    assert validate_float('1') == 1.0
    assert validate_float(1.0, lower=1.0, include_lower=True) == 1.0
    assert validate_float(1.0, upper=1.0, include_upper=True) == 1.0

    with pytest.raises(TypeError):
        validate_float('x')
    with pytest.raises(ValueError):
        validate_float(1.0, lower=2.0)
    with pytest.raises(ValueError):
        validate_float(1.0, upper=0.0)
    with pytest.raises(ValueError):
        validate_float(1.0, upper=1.0, include_upper=False)
    with pytest.raises(ValueError):
        validate_float(1.0, lower=1.0, include_lower=False)
    with pytest.raises(ValueError):
        validate_float(np.inf)


def test_validate_positive_float():

    assert validate_positive_float(1.0) == 1.0
    assert validate_positive_float('1.0') == 1.0

    with pytest.raises(TypeError):
        validate_positive_float('x')
    with pytest.raises(ValueError):
        validate_positive_float(0)
    with pytest.raises(ValueError):
        validate_positive_float(-1.5)


def test_validate_non_negative_float():

    assert validate_non_negative_float(1.0) == 1.0
    assert validate_non_negative_float(0.0) == 0.0
    assert validate_non_negative_float('0.0') == 0.0

    with pytest.raises(TypeError):
        validate_non_negative_float('x')
    with pytest.raises(ValueError):
        validate_non_negative_float(-1.0)


# int ----------------------------------------------------------------------------------------------

def test_validate_int():

    assert validate_float(1.0) == 1
    assert validate_int(1) == 1
    assert validate_int('1') == 1
    assert validate_int(1, lower=1, include_lower=True) == 1
    assert validate_int(1, upper=1, include_upper=True) == 1

    with pytest.raises(TypeError):
        validate_int('x')
    with pytest.raises(ValueError):
        validate_int(1, lower=2)
    with pytest.raises(ValueError):
        validate_int(1, upper=0)
    with pytest.raises(ValueError):
        validate_int(1, upper=1, include_upper=False)
    with pytest.raises(ValueError):
        validate_int(1, lower=1, include_lower=False)
    with pytest.raises(TypeError):
        validate_int(np.inf)


def test_validate_positive_int():

    assert validate_positive_int(1) == 1
    assert validate_positive_int('1') == 1

    with pytest.raises(TypeError):
        validate_positive_int('x')
    with pytest.raises(ValueError):
        validate_positive_int(0)
    with pytest.raises(ValueError):
        validate_positive_int(-1)


def test_validate_non_negative_int():

    assert validate_non_negative_int(1) == 1
    assert validate_non_negative_int(0) == 0
    assert validate_non_negative_int('0') == 0

    with pytest.raises(TypeError):
        validate_non_negative_int('x')
    with pytest.raises(ValueError):
        validate_non_negative_int(-1)

# misc ---------------------------------------------------------------------------------------------


def test_validate_binary():

    assert validate_binary(0) == 0
    assert validate_binary(1) == 1
    assert validate_binary(0.0) == 0
    assert validate_binary(1.0) == 1
    assert validate_binary('0') == 0
    assert validate_binary('1') == 1

    with pytest.raises(TypeError):
        validate_binary('x')
    with pytest.raises(ValueError):
        validate_binary(-1)
    with pytest.raises(ValueError):
        validate_binary(2)
    with pytest.raises(ValueError):
        validate_binary(0, include_lower=False)
    with pytest.raises(ValueError):
        validate_binary(1, include_upper=False)


def test_validate_proportion():

    assert validate_proportion(0.0) == 0.0
    assert validate_proportion(1.0) == 1.0
    assert validate_proportion(0) == 0.0
    assert validate_proportion(1) == 1.0
    assert validate_proportion('0.0') == 0.0
    assert validate_proportion('1.0') == 1.0
    assert validate_proportion(0.0, include_lower=True) == 0.0
    assert validate_proportion(1.0, include_upper=True) == 1.0

    with pytest.raises(TypeError):
        validate_proportion('x')
    with pytest.raises(ValueError):
        validate_proportion(-0.1)
    with pytest.raises(ValueError):
        validate_proportion(1.1)
    with pytest.raises(ValueError):
        validate_proportion(0.0, include_lower=False)
    with pytest.raises(ValueError):
        validate_proportion(1.0, include_upper=False)
