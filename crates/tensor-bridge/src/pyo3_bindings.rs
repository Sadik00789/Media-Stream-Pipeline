use pyo3::prelude::*;
use pyo3::types::PyList;

/// Converts a slice of float32 PCM samples into a Python list of floats.
/// Zero-dependency on external C numpy capsules, runs safely in any Python environment.
pub fn slice_to_pylist<'py>(
    py: Python<'py>,
    slice: &[f32],
) -> Bound<'py, PyList> {
    PyList::new_bound(py, slice)
}

/// Converts a raw pointer and length into a Python list.
/// 
/// # Safety
/// Caller guarantees ptr is non-null, aligned, and valid for `len` elements.
pub unsafe fn raw_ptr_to_pylist<'py>(
    py: Python<'py>,
    ptr: *const f32,
    len: usize,
) -> Bound<'py, PyList> {
    let slice = std::slice::from_raw_parts(ptr, len);
    PyList::new_bound(py, slice)
}
