# earth2bufrio

**BUFR decoder with Python and Fortran backends — reads WMO BUFR and NCEP
PrepBUFR files into PyArrow Tables.**

earth2bufrio decodes binary BUFR (Binary Universal Form for the Representation
of meteorological data) files produced by WMO member agencies and returns the
observations as a wide-format [PyArrow](https://arrow.apache.org/docs/python/)
`Table` with one column per mnemonic, ready for downstream analytics.

## Installation

Install from a local checkout:

```bash
pip install earth2bufrio
```

For development (includes linters and test dependencies):

```bash
pip install -e ".[dev,docs]"
```

To enable the Fortran backend (requires gfortran and CMake):

```bash
make fortran
```

## Quickstart

### Python backend (any WMO BUFR file)

```python
import earth2bufrio

# Read an entire BUFR file — wide format, one column per descriptor
table = earth2bufrio.read_bufr("observations.bufr")
print(table.column_names)
# ['message_type', 'message_index', 'subset_index',
#  'YEAR', 'MNTH', 'DAYS', 'HOUR', 'MINU', 'SECO', ...]

# Filter to a specific data category
marine = earth2bufrio.read_bufr(
    "observations.bufr",
    filters={"data_category": 1},
)

# Select only specific mnemonics
subset = earth2bufrio.read_bufr(
    "observations.bufr",
    mnemonics=["CLATH", "CLONH", "TMBR"],
)

# Speed up large files with parallel decoding
table = earth2bufrio.read_bufr("large_file.bufr", workers=4)
```

### Fortran backend (NCEP BUFR / PrepBUFR)

```python
import earth2bufrio

# Read PrepBUFR with the Fortran backend
table = earth2bufrio.read_bufr(
    "prepbufr.gdas.2024010100",
    backend="fortran",
)

# Read satellite radiance BUFR
table = earth2bufrio.read_bufr(
    "1bamua.gdas.2024010100.bufr",
    backend="fortran",
    mnemonics=["SAID", "CLAT", "CLON", "TMBR", "CHNM"],
)
```

## Contents

```{toctree}
:maxdepth: 2

api
format
backends
```
