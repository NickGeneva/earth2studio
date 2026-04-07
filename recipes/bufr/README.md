# earth2bufr

Pure-Python BUFR decoder that reads WMO BUFR Edition 3/4 files into PyArrow Tables.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from earth2bufr import read_bufr

table = read_bufr("path/to/file.bufr")
print(table.schema)
print(table.to_pandas())
```

## Development

```bash
# Lint
make lint

# Format
make format

# Run tests
make test

# Run tests with coverage
make test-cov
```

## License

Apache-2.0
