#!/bin/bash

pyenv --version
py_version=3.10.16
echo ${py_version}
$(pyenv root)/versions/${py_version}/bin/python -m venv .venv && source .venv/bin/activate && pip list
echo $(which python) && echo $(python --version)
pip install -U pip ipykernel nbformat pandas scikit-learn pytest && pip list
pip freeze > requirements.txt
# pip install -r requirements.txt

# install package
pip install e .

# test package
deactivate
rm -rf test_env && $(pyenv root)/versions/${py_version}/bin/python -m venv test_env && source test_env/bin/activate && pip list
pip install -U pip build twine  pytest pytest-cov && pip list
rm -rf dist && python -m build && twine check dist/*
pip install dist/kidneypy-0.0.2-py3-none-any.whl && pytest --cov=kidneypy
deactivate
rm -rf test_env
source .venv/bin/activate