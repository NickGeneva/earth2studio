# BUFR Format Overview

BUFR (Binary Universal Form for the Representation of meteorological data) is a
WMO standard binary format used worldwide for exchanging weather observations,
satellite retrievals, and model output.  earth2bufr supports **Edition 3** and
**Edition 4** of the BUFR specification.

## File Structure

A BUFR message is composed of six sections:

| Section | Name                  | Contents                                      |
|---------|-----------------------|-----------------------------------------------|
| 0       | Indicator             | Magic bytes `BUFR`, total message length, edition number |
| 1       | Identification        | Originating centre, data category, timestamp, subsets, flags |
| 2       | Optional              | Local-use data (skipped by earth2bufr)       |
| 3       | Data Description      | Unexpanded descriptor sequence that defines the data layout |
| 4       | Data                  | Bit-packed observation values                  |
| 5       | End                   | End marker `7777`                              |

Multiple BUFR messages can be concatenated in a single file.  earth2bufr's
reader stage (`_reader.py`) splits the byte stream by scanning for `BUFR` /
`7777` boundaries.

## Edition 3 vs Edition 4

The two editions differ mainly in Section 1 layout:

| Field                | Edition 3           | Edition 4               |
|----------------------|---------------------|-------------------------|
| Section 1 length     | Fixed 18 bytes      | Variable (≥ 22 bytes)   |
| Year encoding        | Year of century     | Full 4-digit year       |
| Master table version | Single byte         | Same                    |
| Local table version  | Single byte         | Same                    |
| Sub-centre           | Not present         | 2-byte field            |

earth2bufr's section parser (`_section.py`) detects the edition from Section 0
and applies the correct Section 1 layout automatically.

## Descriptor System

BUFR uses **F-X-Y descriptors** — 16-bit integers divided into three fields:

```text
F (2 bits) | X (6 bits) | Y (8 bits)
```

The F field determines the descriptor category:

- **F = 0 — Element descriptor**: look up in **Table B** to get the element
  name, units, scale, reference value, and bit width.
- **F = 1 — Replication operator**: replicate the next *X* descriptors *Y*
  times.  When *Y = 0* the replication count is read from the data stream
  ("delayed replication").
- **F = 2 — Operator descriptor**: modify encoding rules (e.g. change data
  width, add associated fields).
- **F = 3 — Sequence descriptor**: expand via **Table D** into a pre-defined
  list of other descriptors.

earth2bufr's descriptor expansion (`_descriptors.py`) recursively expands all
F = 3 references and builds replication groups, producing a flat list of
`ExpandedDescriptor` and `DelayedReplicationMarker` objects that the decoder
consumes.

## Data Encoding

Each element descriptor carries three encoding parameters from Table B:

| Parameter         | Meaning                                              |
|-------------------|------------------------------------------------------|
| **Reference value** | Offset subtracted before scaling (allows negative values) |
| **Scale**           | Power-of-10 divisor applied after adding reference     |
| **Bit width**       | Number of bits occupied in the data section            |

The physical value is recovered as:

```text
value = (raw_bits + reference_value) / 10^scale
```

A value where all bits are set to 1 is the BUFR **missing-data indicator**.

String fields (unit `CCITT IA5`) are encoded as plain ASCII bytes; missing
strings have all bytes set to `0xFF`.

## PrepBUFR (NCEP Extensions)

NCEP distributes observations in **PrepBUFR** files, which embed WMO Table B/D
definitions inside the file itself as "DX table" messages (data category 11).
earth2bufr detects these messages, extracts the embedded tables, and uses them
for descriptor look-up in subsequent messages.  This allows decoding PrepBUFR
files without shipping external table files.
