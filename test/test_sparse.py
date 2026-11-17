import numpy as np
import pytest

import D4M._sparse as sparse


IS_SCIPY = sparse.selected_splinalg_library == "scipy.sparse"
IS_GRAPHBLAS = sparse.selected_splinalg_library == "python-graphblas"


def matrix(dense):
    dense = np.asarray(dense)
    row, col = np.nonzero(dense)
    return sparse.from_coo(row, col, dense[row, col], shape=dense.shape)


def assert_matrix_equal(actual, expected):
    np.testing.assert_array_equal(sparse.to_array(actual), np.asarray(expected))


def test_backend_metadata_and_matrix_type():
    assert sparse.selected_splinalg_library in sparse.supported_splinalg_libraries
    assert isinstance(matrix([[1]]), sparse.Sparse_Matrix)


def test_init():
    if IS_GRAPHBLAS:
        pytest.xfail("GraphBLAS init does not currently accept dense array-like input")

    actual = sparse.init([[1, 0], [0, 2]])

    assert_matrix_equal(actual, [[1, 0], [0, 2]])
    assert sparse.get_shape(actual) == (2, 2)


@pytest.mark.parametrize("shape,dtype", [(None, None), ((3, 4), float)])
def test_from_coo(shape, dtype):
    actual = sparse.from_coo(
        np.array([1, 0]),
        np.array([2, 1]),
        np.array([3, 4]),
        shape=shape,
        dtype=dtype,
    )

    expected_shape = (2, 3) if shape is None else shape
    expected = np.zeros(expected_shape, dtype=dtype)
    expected[1, 2] = 3
    expected[0, 1] = 4
    assert_matrix_equal(actual, expected)
    assert sparse.get_shape(actual) == expected_shape
    if dtype is not None:
        assert sparse.to_array(actual).dtype == np.dtype(dtype)
    if IS_SCIPY:
        assert actual.format == "coo"
        assert actual.has_canonical_format


def test_from_dense_honors_missing_value_and_dtype():
    actual = sparse.from_dense(
        np.array([[1, -1], [-1, 2]]), missing_value=-1, dtype=float
    )

    assert_matrix_equal(actual, [[1, 0], [0, 2]])
    assert sparse.nnz(actual) == 2
    assert sparse.to_array(actual).dtype == np.dtype(float)


def test_empty():
    actual = sparse.empty()

    assert sparse.get_shape(actual) == (0, 0)
    assert sparse.nnz(actual) == 0
    assert sparse.to_array(actual).shape == (0, 0)


def test_astype():
    actual = sparse.astype(matrix([[1, 0], [0, 2]]), float)

    assert_matrix_equal(actual, [[1.0, 0.0], [0.0, 2.0]])
    assert sparse.to_array(actual).dtype == np.dtype(float)


@pytest.mark.parametrize(
    "converter,scipy_format",
    [(sparse.to_coo, "coo"), (sparse.to_csr, "csr"), (sparse.to_csc, "csc")],
)
def test_format_conversions_and_copy_semantics(converter, scipy_format):
    original = matrix([[1, 0], [0, 2]])
    converted = converter(original, copy=True)

    assert_matrix_equal(converted, [[1, 0], [0, 2]])
    assert converted is not original
    assert converter(converted, copy=False) is converted
    if IS_SCIPY:
        assert converted.format == scipy_format


def test_is_coo_format():
    actual = matrix([[1]])

    assert sparse.is_coo_format(actual)
    if IS_SCIPY:
        assert not sparse.is_coo_format(sparse.to_csr(actual))


def test_to_scipy():
    actual = sparse.to_scipy(matrix([[1, 0], [0, 2]]))

    np.testing.assert_array_equal(actual.toarray(), [[1, 0], [0, 2]])


def test_sum_duplicates_returns_none():
    actual = matrix([[1]])

    assert sparse.sum_duplicates(actual) is None
    assert_matrix_equal(actual, [[1]])


def test_eliminate_zeros_returns_none():
    actual = sparse.from_coo([0, 1], [0, 1], [1, 0], shape=(2, 2))

    assert sparse.eliminate_zeros(actual) is None
    assert_matrix_equal(actual, [[1, 0], [0, 0]])
    assert sparse.nnz(actual) == (1 if IS_SCIPY else 2)


def test_copy_returns_independent_matrix():
    original = matrix([[1, 0], [0, 2]])
    actual = sparse.copy(original)

    assert actual is not original
    assert_matrix_equal(actual, [[1, 0], [0, 2]])


@pytest.mark.parametrize(
    "operation,expected",
    [
        (sparse.add, [[6, 2], [3, 12]]),
        (sparse.matmul, [[5, 16], [15, 32]]),
        (sparse.ewise_mult, [[5, 0], [0, 32]]),
    ],
)
def test_algebra(operation, expected):
    left = matrix([[1, 2], [3, 4]])
    right = matrix([[5, 0], [0, 8]])

    actual = operation(left, right)

    assert_matrix_equal(actual, expected)
    if IS_SCIPY and operation in (sparse.add, sparse.matmul):
        assert actual.format == "coo"
    elif IS_SCIPY:
        assert actual.format == "csr"


@pytest.mark.parametrize("operation", [sparse.add, sparse.matmul, sparse.ewise_mult])
def test_algebra_convert_to_coo_false(operation):
    left = matrix([[1, 2], [3, 4]])
    right = matrix([[5, 0], [0, 8]])

    actual = operation(left, right, convert_to_coo=False)

    assert sparse.get_shape(actual) == (2, 2)
    if IS_SCIPY and operation in (sparse.add, sparse.matmul):
        assert actual.format == "csr"


def test_nnz():
    assert sparse.nnz(matrix([[1, 0], [2, 3]])) == 3


@pytest.mark.parametrize(
    "dimension,expected",
    [(None, 10), (0, [4, 6]), (1, [3, 7])],
)
def test_sparse_sum(dimension, expected):
    if IS_GRAPHBLAS:
        pytest.xfail("GraphBLAS sparse_sum uses APIs removed by the current dependency")

    actual = sparse.sparse_sum(matrix([[1, 2], [3, 4]]), dimension)

    if dimension is None:
        assert actual == expected
    else:
        np.testing.assert_array_equal(np.asarray(actual), expected)


def test_logical_copy_semantics():
    original = matrix([[2, 0], [0, 3]])
    actual = sparse.logical(original, copy=True)

    assert_matrix_equal(actual, [[1, 0], [0, 1]])
    assert_matrix_equal(original, [[2, 0], [0, 3]])


def test_transpose_copy_semantics():
    original = matrix([[1, 2], [0, 3]])
    actual = sparse.transpose(original, copy=True)

    assert_matrix_equal(actual, [[1, 0], [2, 3]])
    assert_matrix_equal(original, [[1, 2], [0, 3]])


def test_coo_canonicalize_copy_semantics():
    original = matrix([[1, 0], [0, 2]])
    actual = sparse.coo_canonicalize(original, copy=True)

    assert actual is not original
    assert_matrix_equal(actual, [[1, 0], [0, 2]])
    if IS_SCIPY:
        assert actual.format == "coo"
        assert actual.has_canonical_format


def test_sparse_equal_value_shape_and_tolerance():
    actual = matrix([[1.0, 0.0], [0.0, 2.0]])

    assert sparse.sparse_equal(actual, matrix([[1.0, 0.0], [0.0, 2.0]]))
    assert sparse.sparse_equal(actual, matrix([[1.0, 0.0], [0.0, 2.000001]]))
    assert not sparse.sparse_equal(actual, matrix([[1.0, 0.0], [0.0, 3.0]]))
    assert not sparse.sparse_equal(actual, matrix([[1.0, 0.0, 0.0]]))


def test_get_data_dtype_and_shape():
    actual = sparse.from_coo([1, 0], [0, 1], [3, 2], shape=(2, 3), dtype=float)

    np.testing.assert_array_equal(sparse.get_data(actual), [2.0, 3.0])
    assert sparse.get_dtype(actual) == np.dtype(float)
    assert sparse.get_shape(actual) == (2, 3)


def test_get_coo_params_order_and_canonicalization():
    actual = sparse.from_coo([1, 0], [0, 1], [3, 2], shape=(2, 2))

    row, col, data = sparse.get_coo_params(actual)

    np.testing.assert_array_equal(row, [0, 1])
    np.testing.assert_array_equal(col, [1, 0])
    np.testing.assert_array_equal(data, [2, 3])


def test_get_csr_params_order():
    actual = matrix([[1, 0, 2], [0, 3, 0]])

    data, indices, indptr = sparse.get_csr_params(actual)

    np.testing.assert_array_equal(data, [1, 2, 3])
    np.testing.assert_array_equal(indices, [0, 2, 1])
    np.testing.assert_array_equal(indptr, [0, 2, 3])


def test_get_csc_params_order():
    actual = matrix([[1, 0, 2], [0, 3, 0]])

    data, indices, indptr = sparse.get_csc_params(actual)

    np.testing.assert_array_equal(data, [1, 3, 2])
    np.testing.assert_array_equal(indices, [0, 1, 0])
    np.testing.assert_array_equal(indptr, [0, 1, 2, 3])


@pytest.mark.parametrize(
    "selector,selection,expected",
    [
        (sparse.getitem_rows, [True, False, True], [[1, 2], [5, 6]]),
        (sparse.getitem_cols, [False, True], [[2], [4], [6]]),
    ],
)
def test_boolean_selection(selector, selection, expected):
    actual = selector(matrix([[1, 2], [3, 4], [5, 6]]), selection)

    assert_matrix_equal(actual, expected)


def test_getvalue_present_and_missing():
    actual = matrix([[1, 0], [0, 2]])

    assert sparse.getvalue(actual, 1, 1) == 2
    assert sparse.getvalue(actual, 0, 1) == (0 if IS_SCIPY else None)


def test_update_data():
    if IS_GRAPHBLAS:
        pytest.xfail("GraphBLAS update_data currently misorders COO values and indices")

    original = sparse.to_csr(matrix([[1, 0], [0, 2]]))
    actual = sparse.update_data(original, np.array([3, 4]))

    assert_matrix_equal(actual, [[3, 0], [0, 4]])
    assert actual.format == "csr"
