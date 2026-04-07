.PHONY: lint format test test-cov docs fortran rust

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=earth2bufrio --cov-report=term-missing --cov-fail-under=80

docs:
	sphinx-build -b html docs/ docs/_build/html

fortran:
	cmake -S src/fortran -B build/fortran -DCMAKE_BUILD_TYPE=Release
	cmake --build build/fortran --parallel
	cp build/fortran/libearth2bufrio_fort.so src/earth2bufrio/

rust:
	cd src/rust && maturin develop --release
