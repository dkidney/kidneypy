# kidneypy

Short description of what the package does and the problem it solves.

![License](https://img.shields.io/github/license/dkidney/kidneypy)
![CI](https://github.com/dkidney/kidneypy/actions/workflows/ci.yml/badge.svg)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-teal.json)](https://github.com/copier-org/copier)

## Features

## Installation

### Requirements

- Python 3.12
- uv

```sh
brew install uv
```

### Install from GitHub

#### Install into an environment

From a specific release tag:

```sh
uv add "git+https://github.com/dkidney/kidneypy.git@v0.1.0"
```

From a specific branch:

```sh
uv add "git+https://github.com/dkidney/kidneypy.git@develop"
```

#### Add as a dependency in your pyproject.toml file

```toml
[project]
dependencies = [
    "kidneypy",
]

From a specific release tag:

```toml
[tool.uv.sources]
kidneypy = { git = "https://github.com/dkidney/kidneypy.git", tag = "v0.1.0" }
```

From a specific branch:

```toml
[tool.uv.sources]
kidneypy = { git = "https://github.com/dkidney/kidneypy.git", branch = "develop" }
```

## Quick Start

A minimal, working example showing how to import and use the package.

```python
from kidneypy import helloworld

print(helloworld())
```

## Usage

More detailed examples covering the package's main use cases.

### Basic Usage

### Advanced Usage

### Configuration

## API Reference

Description of the main public classes, functions, and modules, or a link to generated API documentation.

## Examples

Practical examples, scripts, notebooks, or links to example files.

## Development

Instructions for setting up a local development environment.

### Set Up the Environment

### Run Tests

### Code Quality and Formatting

## Contributing

Guidelines for reporting issues, submitting pull requests, and contributing code.

## License

State the license and link to the relevant `LICENSE` file.

## Acknowledgements

Credit to relevant projects, contributors, or funding sources, if applicable.
