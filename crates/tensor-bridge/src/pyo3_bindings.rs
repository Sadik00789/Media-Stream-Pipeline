use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use numpy::{PyArray1, PyArrayMethods};

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
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    PyList::new_bound(py, slice)
}

/// Returns a zero-copy PyArray3<u8> of shape [480, 640, 3] mapped directly to pixel_data.
pub fn as_pyarray_video<'py>(
    py: Python<'py>,
    pixel_data: &[u8],
) -> PyResult<Bound<'py, numpy::PyArray3<u8>>> {
    let arr1d = PyArray1::from_slice_bound(py, pixel_data);
    let reshaped = arr1d.reshape([480, 640, 3])?;
    reshaped.extract()
}

/// Helper converting raw video bytes into a Python bytes object
pub fn slice_to_pybytes<'py>(
    py: Python<'py>,
    slice: &[u8],
) -> Bound<'py, PyBytes> {
    PyBytes::new_bound(py, slice)
}
