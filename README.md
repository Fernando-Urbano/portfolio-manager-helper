# Portfolio Manager Package

## Create Virtual Environment
The virtual environment is created using conda. It is used to manage the dependencies of the package.

The following steps are used to create the virtual environment.

Inside your terminal, write:
```bash
conda create -n finm python=3.12.5
```

Activate virtual environment:
```bash
conda activate finm
```

Install packages:
```bash
pip install -r requirements-dev.txt
```

Install pre-commit hooks:
```bash
pre-commit install
```

## How to Update version in Pipy?

Update the `version/new_version.txt` file with the new version number.

Install the following packages:

```bash
pip install --upgrade pip setuptools wheel
```

```bash
pip install --upgrade twine
```

Remove existing `build/` and `dist/` folders to ensure a clean build:
```bash
rm -rf build/ dist/ *.egg-info
```

From the project root directory, run:

```bash
python setup.py sdist bdist_wheel
```

Upload the package to pip wit the new version:

```bash
twine upload dist/* -u __token__ -p $(cat build-package/token.txt)
```

Check that the version of the package is updated in the [PyPi](https://pypi.org/project/portfolio-management/).
