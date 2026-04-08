.PHONY: lint format test test-cov docs fortran rust install-test setup

# ── Linting & formatting ────────────────────────────────────────────
lint:
	ruff check src/ test/

format:
	ruff format src/ test/

# ── Dependencies ────────────────────────────────────────────────────
install-test:
	uv pip install -e ".[test]"

# ── Native backends ─────────────────────────────────────────────────
fortran:
	@# Clone NCEPLIBS-bufr if not present
	@if [ ! -d src/fortran/nceplibs-bufr ]; then \
		echo "Cloning NCEPLIBS-bufr..."; \
		git clone --depth 1 https://github.com/NOAA-EMC/NCEPLIBS-bufr.git \
			src/fortran/nceplibs-bufr; \
	fi
	cmake -S src/fortran -B build/fortran \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=build/fortran/install \
		-DBUILD_TESTING=OFF \
		-DENABLE_DOCS=OFF
	cmake --build build/fortran --parallel
	cp build/fortran/libearth2bufr_fort.so src/earth2bufr/

rust:
	cd src/rust && maturin build --release -o ../../build/rust
	@# Extract the compiled extension from the wheel into earth2bufr
	python -c "import zipfile, glob; \
		whl = glob.glob('build/rust/earth2bufr-*.whl')[0]; \
		zf = zipfile.ZipFile(whl); \
		so = [n for n in zf.namelist() if n.endswith('.so')][0]; \
		data = zf.read(so); \
		out = 'src/earth2bufr/_lib' + so[so.rfind('.cpython'):]; \
		open(out, 'wb').write(data); \
		print(f'Copied {so} -> {out}')"

# ── Full setup (deps + native backends) ─────────────────────────────
setup: install-test fortran rust

# ── Tests ───────────────────────────────────────────────────────────
test: setup
	pytest test/ -v --timeout=120

test-cov: setup
	pytest test/ -v --timeout=120 --cov=earth2bufr --cov-report=term-missing --cov-fail-under=80

# ── Docs ────────────────────────────────────────────────────────────
docs:
	sphinx-build -b html docs/ docs/_build/html
