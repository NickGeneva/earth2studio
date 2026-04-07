<!-- markdownlint-disable MD013 -->

# earth2bufrio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development or
> superpowers:executing-plans.

**Goal:** Build a pure-Python BUFR decoder that reads Edition
3/4 files into PyArrow Tables.

**Architecture:** 5-stage pipeline
(reader -> section -> descriptors -> decoder -> arrow) with
maturin build skeleton for future Rust backend.

**Tech Stack:** Python 3.11+, PyArrow, maturin, ruff, pytest,
Sphinx

---

## File Map

All paths relative to `recipes/earth2bufrio/`.

**Package sources** (`src/earth2bufrio/`):

- `__init__.py` — re-export `read_bufr`, `BufrDecodeError`
- `_types.py` — dataclasses + exceptions
- `_tables.py` — WMO Table B/D loading + DX table parsing
- `tables/table_b.json` — bundled WMO Table B
- `tables/table_d.json` — bundled WMO Table D
- `_reader.py` — binary message splitting
- `_section.py` — section parsing (Ed3/Ed4)
- `_descriptors.py` — F-X-Y expansion, replication, operators
- `_decoder.py` — bit-level data decoding
- `_arrow.py` — PyArrow Table construction
- `_api.py` — `read_bufr()` public function

**Rust skeleton** (`src/rust/`):

- `Cargo.toml` + `src/lib.rs` — empty PyO3 module

**Tests** (`tests/`):

- `conftest.py`, `test_types.py`, `test_tables.py`,
  `test_reader.py`, `test_section.py`, `test_descriptors.py`,
  `test_decoder.py`, `test_arrow.py`, `test_api.py`,
  `test_crossval.py`, `generate_references.py`
- `data/*.bufr` + `data/*.ref.json` — fixtures

**Docs** (`docs/`):

- `conf.py`, `index.md`, `api.md`, `format.md`, `backends.md`

**Root files:**

- `pyproject.toml`, `Makefile`, `README.md`

---

### Task 1: Project Scaffold

**Files:** Create `pyproject.toml`, `Makefile`, `README.md`,
`src/earth2bufrio/__init__.py`, `src/rust/Cargo.toml`,
`src/rust/src/lib.rs`

- [ ] **1.1** Create `pyproject.toml` with maturin build,
  pyarrow dep, ruff/pytest/coverage/interrogate config.
  Use physicsnemo-curator as reference (line-length=120,
  py311, numpy docstrings, 80% coverage, 99% interrogate).

- [ ] **1.2** Create empty `src/rust/Cargo.toml` and
  `src/rust/src/lib.rs` (PyO3 skeleton that exposes
  `earth2bufrio._lib` module with no functions).

- [ ] **1.3** Create `Makefile` with targets: `lint`,
  `format`, `test`, `test-cov`, `docs`.

- [ ] **1.4** Create `README.md` with package description,
  install instructions, quick usage example.

- [ ] **1.5** Create `src/earth2bufrio/__init__.py` that
  re-exports `read_bufr` and `BufrDecodeError` (stub
  imports for now).

- [ ] **1.6** Verify: `maturin develop` succeeds,
  `python -c "import earth2bufrio"` works.

- [ ] **1.7** Commit: `feat: scaffold earth2bufrio package`

---

### Task 2: Types & Exceptions (`_types.py`)

**Files:** Create `src/earth2bufrio/_types.py`,
`tests/conftest.py`, `tests/test_types.py`

- [ ] **2.1** Write `tests/conftest.py` with pytest markers
  (unit, crossval, integration, slow) and `DATA_DIR`
  fixture pointing to `tests/data/`.

- [ ] **2.2** Write `tests/test_types.py` — test that
  dataclasses instantiate correctly and `BufrDecodeError`
  is a subclass of `Exception`.

- [ ] **2.3** Run tests, verify they fail.

- [ ] **2.4** Implement `_types.py` with frozen slotted
  dataclasses:
  - `BufrDecodeError(Exception)` — message + optional offset
  - `BufrMessage` — data, offset, index
  - `IndicatorSection` — length, edition
  - `IdentificationSection` — originating_center,
    data_category, year, month, day, hour, minute, second,
    num_subsets, observed, compressed
  - `DataDescriptionSection` — descriptors: list[int]
  - `ParsedMessage` — indicator, identification,
    data_description, data_bytes
  - `TableBEntry` — name, units, scale, reference_value,
    bit_width
  - `TableDEntry` — descriptors: list[int]
  - `ExpandedDescriptor` — id, entry (TableBEntry)
  - `DecodedSubset` — values: list of (descriptor, value)

- [ ] **2.5** Run tests, verify pass.

- [ ] **2.6** Commit: `feat: add BUFR dataclasses and
  exceptions`

---

### Task 3: Table Management (`_tables.py`)

**Files:** Create `src/earth2bufrio/_tables.py`,
`src/earth2bufrio/tables/table_b.json`,
`src/earth2bufrio/tables/table_d.json`,
`tests/test_tables.py`

- [ ] **3.1** Write a script to copy and reformat
  pybufrkit's Table B/D JSON into our bundled format.
  Source: pybufrkit `tables/0/0_0/12/`.
  Our Table B format:
  `{"FXXYYY": {"name", "units", "scale", "reference_value",
  "bit_width"}}`.
  Table D format: `{"FXXYYY": [int, ...]}`.

- [ ] **3.2** Write `tests/test_tables.py`:
  - `load_table_b()` has `012001` (Temperature)
  - `load_table_d()` has `301011` ([4001, 4002, 4003])
  - `TableSet.lookup_b(12001)` returns `TableBEntry`
  - `TableSet.lookup_d(301011)` returns list of ints
  - `TableSet.add_b()` overrides WMO entries

- [ ] **3.3** Run tests, verify fail.

- [ ] **3.4** Implement `_tables.py`:
  - `load_table_b()` / `load_table_d()` via
    `importlib.resources`
  - `class TableSet` with lookup_b, lookup_d, add_b, add_d
  - `parse_dx_table()` — stub returning empty dict

- [ ] **3.5** Run tests, verify pass.

- [ ] **3.6** Commit:
  `feat: add WMO table management and bundled tables`

---

### Task 4: Message Reader (`_reader.py`)

**Files:** Create `src/earth2bufrio/_reader.py`,
`tests/test_reader.py`

- [ ] **4.1** Write `tests/test_reader.py`:
  - Single valid message yields one `BufrMessage`
  - Two concatenated messages yield two
  - Empty bytes yields nothing
  - No BUFR marker yields nothing
  - Truncated message raises `BufrDecodeError`
  - Bad `7777` end marker raises `BufrDecodeError`

  Hand-craft minimal binary: `b"BUFR"` + 3-byte length
  (big-endian, value=12) + 1-byte edition (4) + 4 zero
  bytes + `b"7777"` = 12 bytes total.

- [ ] **4.2** Run tests, verify fail.

- [ ] **4.3** Implement `_reader.py`:
  `read_messages(data: bytes) -> Iterator[BufrMessage]` —
  scan for `b"BUFR"`, read 3-byte length, extract slice,
  validate `7777` trailer, yield `BufrMessage`.

- [ ] **4.4** Run tests, verify pass.

- [ ] **4.5** Commit: `feat: add BUFR message reader`

---

### Task 5: Section Parsing (`_section.py`)

**Files:** Create `src/earth2bufrio/_section.py`,
`tests/test_section.py`

- [ ] **5.1** Write `tests/test_section.py`:
  - Ed4 identification section (22-byte)
  - Ed3 identification section (18-byte, 1-byte year)
  - Data description section extracts descriptor list
  - `parse_message()` returns full `ParsedMessage`

  Hand-craft binary sections per BUFR spec.

- [ ] **5.2** Run tests, verify fail.

- [ ] **5.3** Implement `_section.py`:
  - `parse_message(msg) -> ParsedMessage`
  - `_parse_indicator(data)` — bytes 0-3 BUFR, 4-6 length,
    7 edition
  - `_parse_identification_ed4(data, offset)` — 2-byte year
  - `_parse_identification_ed3(data, offset)` — 1-byte year
  - `_parse_data_description(data, offset)` — 2-byte
    descriptor list

- [ ] **5.4** Run tests, verify pass.

- [ ] **5.5** Commit:
  `feat: add BUFR section parser (Ed3/Ed4)`

---

### Task 6: Descriptor Expansion (`_descriptors.py`)

**Files:** Create `src/earth2bufrio/_descriptors.py`,
`tests/test_descriptors.py`

- [ ] **6.1** Write `tests/test_descriptors.py`:
  - F=0 (Table B) -> single ExpandedDescriptor
  - F=3 (Table D) -> expanded sequence
  - Nested Table D -> fully flattened
  - F=1 replication (regular + delayed)
  - F=2 operators (201YYY change width, 202YYY change scale)

- [ ] **6.2** Run tests, verify fail.

- [ ] **6.3** Implement `_descriptors.py`:
  `expand_descriptors(raw_ids, tables) -> list[ExpandedDescriptor]`
  - F extraction: `F = id // 100000`
  - F=0: Table B lookup
  - F=3: Table D recursive expand
  - F=1: replication (X=count, X=0 delayed)
  - F=2: operator state tracking
  - Max depth 50 guard

- [ ] **6.4** Run tests, verify pass.

- [ ] **6.5** Commit:
  `feat: add descriptor expansion with replication and operators`

---

### Task 7: Bit-Level Decoder (`_decoder.py`)

**Files:** Create `src/earth2bufrio/_decoder.py`,
`tests/test_decoder.py`

- [ ] **7.1** Write `tests/test_decoder.py`:
  - `_read_bits()` extracts correct unsigned integer
  - Uncompressed: 1 subset, 2 descriptors
  - Compressed: 2 subsets, 1 descriptor
  - Missing value (all bits=1 -> None)
  - String decoding (CCITT IA5)
  - Decode formula: `(raw + ref) / 10^scale`

  Hand-craft bit sequences for each test.

- [ ] **7.2** Run tests, verify fail.

- [ ] **7.3** Implement `_decoder.py`:
  - `_read_bits(data, bit_offset, num_bits) -> int`
  - `_is_missing(raw, num_bits) -> bool`
  - `_decode_value(raw, entry) -> float | None`
  - `_decode_string(data, bit_offset, num_bytes) -> str`
  - `decode(expanded, data_bytes, num_subsets, compressed)`
    - Uncompressed: sequential subset iteration
    - Compressed: min + 6-bit increment width + per-subset
    - Delayed replication: read count from stream

- [ ] **7.4** Run tests, verify pass.

- [ ] **7.5** Commit: `feat: add bit-level BUFR decoder`

---

### Task 8: Arrow Table Construction (`_arrow.py`)

**Files:** Create `src/earth2bufrio/_arrow.py`,
`tests/test_arrow.py`

- [ ] **8.1** Write `tests/test_arrow.py`:
  - `build_table()` with mock data -> correct schema + rows
  - Column filtering
  - Well-known descriptor promotion (lat, lon, time,
    station_id, pressure, elevation)
  - Quality mark extraction
  - Empty input -> empty table with correct schema

- [ ] **8.2** Run tests, verify fail.

- [ ] **8.3** Implement `_arrow.py`:
  - Output schema per spec (14 columns)
  - Well-known descriptor ID sets
  - `build_table(decoded, columns) -> pa.Table`
  - Promote known descriptors to named columns
  - Long-format rows for remaining descriptors

- [ ] **8.4** Run tests, verify pass.

- [ ] **8.5** Commit: `feat: add PyArrow table construction`

---

### Task 9: Public API (`_api.py`) & `__init__.py`

**Files:** Create `src/earth2bufrio/_api.py`, update
`src/earth2bufrio/__init__.py`, `tests/test_api.py`

- [ ] **9.1** Write `tests/test_api.py`:
  - Nonexistent path raises `FileNotFoundError`
  - Empty file -> empty table with correct schema
  - Hand-crafted BUFR -> expected rows
  - `filters={"data_category": 0}` skips other messages
  - `columns=["value"]` -> single-column table
  - `workers=2` same result as `workers=1`

- [ ] **9.2** Run tests, verify fail.

- [ ] **9.3** Implement `_api.py` — `read_bufr()`:
  1. Read file bytes
  2. Split messages
  3. Extract DX tables (data_category=11)
  4. Filter messages
  5. Expand descriptors + decode
  6. Build Arrow table
  7. For workers>1 use ProcessPoolExecutor

- [ ] **9.4** Update `__init__.py` imports.

- [ ] **9.5** Run tests, verify pass.

- [ ] **9.6** Commit: `feat: add read_bufr() public API`

---

### Task 10: Test Fixtures & Cross-Validation

**Files:** Create `tests/data/`, `tests/generate_references.py`,
`tests/test_crossval.py`

- [ ] **10.1** Download 5 BUFR fixtures from pybufrkit GitHub:
  `profiler_european.bufr`, `207003.bufr`, `uegabe.bufr`,
  `g2nd_208.bufr`, `b005_89.bufr`

- [ ] **10.2** Write `tests/generate_references.py` using
  pybufrkit to create `.ref.json` reference files.

- [ ] **10.3** Run generator to create `.ref.json` files.

- [ ] **10.4** Write `tests/test_crossval.py`:
  parametrized over all 5 fixtures, compare against
  `.ref.json` (numeric tol 1e-6, exact string/None).
  Mark with `@pytest.mark.crossval`.

- [ ] **10.5** Run crossval tests, iterate until passing.

- [ ] **10.6** Commit:
  `test: add cross-validation fixtures and tests`

---

### Task 11: Documentation

**Files:** Create `docs/conf.py`, `docs/index.md`,
`docs/api.md`, `docs/format.md`, `docs/backends.md`

- [ ] **11.1** `docs/conf.py` — Sphinx 9,
  nvidia-sphinx-theme, MyST, sphinx-autoapi, numpydoc.

- [ ] **11.2** `docs/index.md` — landing page with install,
  quickstart, toctree.

- [ ] **11.3** `docs/api.md` — API reference stub.

- [ ] **11.4** `docs/format.md` — BUFR format overview.

- [ ] **11.5** `docs/backends.md` — backend architecture.

- [ ] **11.6** Verify `make docs` builds without errors.

- [ ] **11.7** Commit: `docs: add Sphinx documentation`

---

### Task 12: Final Validation

- [ ] **12.1** Run full test suite with coverage:
  `pytest tests/ -v --cov=earth2bufrio --cov-fail-under=80`

- [ ] **12.2** Run linter: `ruff check src/ tests/`

- [ ] **12.3** Run formatter: `ruff format --check src/ tests/`

- [ ] **12.4** Run interrogate:
  `interrogate -v src/earth2bufrio/` (99% threshold)

- [ ] **12.5** Run type checker: `ty check src/`

- [ ] **12.6** Fix any failures.

- [ ] **12.7** Commit: `chore: final lint and coverage fixes`
