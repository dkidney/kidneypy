import pytest
from pydantic import ValidationError
from kidneypy.df import (
    Profiler,
    generate_example_df,
)

df = generate_example_df(n=1000)
profile = Profiler(df)

def test_validation_max_p_missing():
    profile.clean(dry_run=True, max_p_missing=0)
    profile.clean(dry_run=True, max_p_missing=0.5)
    profile.clean(dry_run=True, max_p_missing=1.0)
    with pytest.raises(ValidationError):
        profile.clean(dry_run=True, max_p_missing=-1)
    with pytest.raises(ValidationError):
        profile.clean(dry_run=True, max_p_missing=2)


def test_validation_max_cardinality():
    profile.clean(dry_run=True, max_cardinality=1)
    profile.clean(dry_run=True, max_cardinality=10)
    with pytest.raises(ValidationError):
        profile.clean(dry_run=True, max_cardinality=0)
    with pytest.raises(ValidationError):
        profile.clean(dry_run=True, max_cardinality=-1)


def test_validation_min_std():
    profile.clean(dry_run=True, min_std=1)
    profile.clean(dry_run=True, min_std=0.01)
    with pytest.raises(ValidationError):
        profile.clean(dry_run=True, min_std=0)
    with pytest.raises(ValidationError):
        profile.clean(dry_run=True, min_std=-1)


def test_validation_max_p_mode():
    profile.clean(dry_run=True, max_p_mode=1)
    profile.clean(dry_run=True, max_p_mode=0.5)
    with pytest.raises(ValidationError):
        profile.clean(dry_run=True, max_p_mode=0)
    with pytest.raises(ValidationError):
        profile.clean(dry_run=True, max_p_mode=2)
    with pytest.raises(ValidationError):
        profile.clean(dry_run=True, max_p_mode=-1)
