# API Reference

The public API of earth2bufrio consists of a single entry-point function and one
exception class.  Full signatures and docstrings are generated automatically by
[sphinx-autoapi](https://sphinx-autoapi.readthedocs.io/) from the source code.

## `read_bufr`

The primary function for decoding BUFR files:

```python
import earth2bufrio

table = earth2bufrio.read_bufr(
    "observations.bufr",
    columns=None,                    # Optional list of column names
    filters={"data_category": 102},  # Optional message-level filters
    workers=1,                       # >1 enables multiprocess decoding
)
```

`read_bufr` returns a `pyarrow.Table` in long format with the following columns:

| Column             | Type      | Description                                    |
|--------------------|-----------|------------------------------------------------|
| `message_index`    | `int64`   | Zero-based index of the BUFR message in the file |
| `subset_index`     | `int64`   | Zero-based index of the subset within the message |
| `data_category`    | `int64`   | BUFR Table A data category code                |
| `latitude`         | `float64` | Observation latitude (degrees north)           |
| `longitude`        | `float64` | Observation longitude (degrees east)           |
| `time`             | `string`  | ISO-8601 observation timestamp                 |
| `station_id`       | `string`  | Station or platform identifier                 |
| `pressure`         | `float64` | Pressure level (Pa)                            |
| `elevation`        | `float64` | Station elevation (m)                          |
| `descriptor_id`    | `string`  | FXY descriptor in ``"FXXYYY"`` notation        |
| `descriptor_name`  | `string`  | Human-readable element name from Table B       |
| `value`            | `float64` | Decoded physical value                         |
| `units`            | `string`  | Unit string from Table B                       |
| `quality_mark`     | `float64` | Quality indicator (when available)             |

## `BufrDecodeError`

Raised when a BUFR message cannot be decoded — for example when the file is
truncated, contains an unsupported edition, or has malformed section headers.

```python
from earth2bufrio import BufrDecodeError

try:
    table = earth2bufrio.read_bufr("corrupt.bufr")
except BufrDecodeError as exc:
    print(f"Decode failed at byte {exc.offset}: {exc}")
```

## Auto-generated API

The sections below are produced by sphinx-autoapi from the package source.
Internal modules (prefixed with `_`) are excluded except where they define
public symbols re-exported by `earth2bufrio.__init__`.
