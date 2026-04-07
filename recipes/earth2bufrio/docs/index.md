# earth2bufrio

**Pure-Python BUFR decoder — reads WMO BUFR Edition 3/4 files into PyArrow Tables.**

earth2bufrio decodes binary BUFR (Binary Universal Form for the Representation
of meteorological data) files produced by WMO member agencies and returns the
observations as a flat [PyArrow](https://arrow.apache.org/docs/python/) `Table`
with 14 columns ready for downstream analytics.

## Installation

Install from a local checkout:

```bash
pip install earth2bufrio
```

For development (includes linters and test dependencies):

```bash
pip install -e ".[dev,docs]"
```

## Quickstart

```python
import earth2bufrio

# Read an entire BUFR file
table = earth2bufrio.read_bufr("observations.bufr")
print(table.schema.names)
# ['message_index', 'subset_index', 'data_category', 'latitude', 'longitude',
#  'time', 'station_id', 'pressure', 'elevation', 'descriptor_id',
#  'descriptor_name', 'value', 'units', 'quality_mark']

# Filter to a specific data category (e.g. surface marine)
marine = earth2bufrio.read_bufr(
    "observations.bufr",
    filters={"data_category": 1},
)

# Select only a subset of columns
subset = earth2bufrio.read_bufr(
    "observations.bufr",
    columns=["latitude", "longitude", "value", "units"],
)

# Speed up large files with parallel decoding
table = earth2bufrio.read_bufr("large_file.bufr", workers=4)
```

## Contents

```{toctree}
:maxdepth: 2

api
format
backends
```
