use pyo3::prelude::*;

/// Placeholder module — will contain Rust BUFR decoder backend.
#[pymodule]
fn _lib(_py: Python, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
