use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

/// Converts a slice of float32 PCM samples into a Python list of floats.
/// Zero-dependency on external C numpy capsules, runs safely in any Python environment.
pub fn slice_to_pylist<'py>(py: Python<'py>, slice: &[f32]) -> Bound<'py, PyList> {
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
pub fn slice_to_pybytes<'py>(py: Python<'py>, slice: &[u8]) -> Bound<'py, PyBytes> {
    PyBytes::new_bound(py, slice)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyUntypedArrayMethods;
    use pyo3::types::PyAnyMethods;

    #[test]
    fn test_pyo3_slice_and_video_array_bindings() {
        Python::with_gil(|py| {
            let pcm = [0.25f32, -0.5f32, 0.75f32];
            let list = slice_to_pylist(py, &pcm);
            assert_eq!(list.len(), 3);
            let first: f32 = list.get_item(0).unwrap().extract().unwrap();
            assert!((first - 0.25).abs() < 1e-6);

            let raw_list = unsafe { raw_ptr_to_pylist(py, pcm.as_ptr(), pcm.len()) };
            assert_eq!(raw_list.len(), 3);

            let bytes = slice_to_pybytes(py, &[1u8, 2, 3, 4]);
            assert_eq!(bytes.as_bytes(), &[1u8, 2, 3, 4]);

            let pixels = vec![128u8; 640 * 480 * 3];
            let arr = as_pyarray_video(py, &pixels).expect("PyArray3 reshape failed");
            assert_eq!(arr.shape(), [480, 640, 3]);
        });
    }
}
