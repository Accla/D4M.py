import numpy as np
from typing import Union, Tuple, Optional
import graphblas as gb
import scipy.sparse as scisparse

supported_splinalg_libraries = {"scipy.sparse", "python-graphblas"}
selected_splinalg_library = "scipy.sparse"  # change to supported sparse linear algebra of choice


def _unsupported():
    raise Exception("A supported sparse linear algebra library must be selected.")


if selected_splinalg_library == "python-graphblas":
    Sparse_Matrix = gb.Matrix
elif selected_splinalg_library == "scipy.sparse":
    Sparse_Matrix = Union[scisparse.coo_array, scisparse.csr_array, scisparse.csc_array, scisparse.sparray]
else:
    _unsupported()
    Sparse_Matrix = None


# Initialization, conversion, and canonicalization methods
def init(sparse_like, *args, **kwargs) -> Sparse_Matrix:
    try:
        if selected_splinalg_library == "python-graphblas":
            sparse_mat = gb.Matrix(sparse_like, *args, **kwargs)
        elif selected_splinalg_library == "scipy.sparse":
            sparse_mat = scisparse.coo_array(sparse_like, *args, **kwargs)
        else:
            sparse_mat = empty()
        return sparse_mat
    except ValueError as e:
        raise e


def from_coo(row: np.ndarray,
             col: np.ndarray,
             val: np.ndarray,
             shape: Optional[Tuple[int, int]] = None,
             dtype: Optional[Union[np.dtype, type[float], type[int]]] = None
             ) \
        -> Sparse_Matrix:
    """
    Initialize a sparse matrix using COOrdinate format.
    :param row: numpy array of row indices
    :param col: numpy array of column indices
    :param val: numpy array of values OR a single value that is appropriately broadcast
    :param shape: Optional 2-tuple of integers indicating the desired shape of the initialized sparse matrix,
        default is None, in which case the appropriate shape is intuited from the row and col parameters
    :param dtype: Optional dtype to initialize the sparse matrix with, default is None, in which case the appropriate
        dtype is intuited from the val parameter
    :return: sparse_mat: the initialized sparse matrix defined by the COOrdinate format inputs row, col, val
    Notes:
        * row and col are assumed to be of equal size
        * if val is not a single value then its size is assumed to be equal to that of row and col
        * When scipy.sparse is the selected sparse linear algebra library, the default format for the initialized
            sparse matrix is COOrdinate in canonical format (i.e., so that values are sorted by row and then by column.
        * No guarantees are provided that the provided dtype parameter be respected by the underlying sparse linear
            algebra library
    """
    if selected_splinalg_library == "python-graphblas":
        shape = (None, None) if shape is None else shape
        sparse_mat = gb.Matrix.from_coo(row, col, val, nrows=shape[0], ncols=shape[1], dtype=dtype)
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat = scisparse.coo_array((val, (row, col)), shape=shape, dtype=dtype)
        sparse_mat = sparse_mat.tocsr().tocoo()
    else:
        sparse_mat = None
    return sparse_mat


def from_dense(np_mat: np.ndarray,
               missing_value: Union[int, float] = 0,
               dtype: Optional[Union[np.dtype, type[float], type[int]]] = None
               ) -> Sparse_Matrix:
    if selected_splinalg_library == "scipy.sparse":
        sparse_mat_out = scisparse.coo_array(np_mat, dtype=dtype)
        if missing_value == 0:
            sparse_mat_out.eliminate_zeros()
        else:
            # shift values so missing_value->0, then eliminate zeroes, then shift values back
            sparse_mat_out.data += -missing_value
            sparse_mat_out.eliminate_zeros()
            sparse_mat_out.data += missing_value
    elif selected_splinalg_library == "python-graphblas":
        sparse_mat_out = gb.Matrix.from_dense(np_mat, missing_value=missing_value, dtype=dtype)
    else:
        sparse_mat_out = empty()
    return sparse_mat_out


def empty() -> Sparse_Matrix:
    """
    Output an empty sparse matrix
    :return: sparse_mat_empty - a sparse matrix with no stored values and of shape (0, 0)
    """
    if selected_splinalg_library == "python-graphblas":
        sparse_mat_empty = gb.Matrix(gb.dtypes.FP64, nrows=0, ncols=0)
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat_empty = scisparse.coo_matrix(([], ([], [])), shape=(0, 0))
    else:
        sparse_mat_empty = "something went wrong"
    return sparse_mat_empty


def astype(sparse_mat: Sparse_Matrix,
           dtype: Union[np.dtype, type(int), type(float)]
           ) \
        -> Sparse_Matrix:
    """
    Change the existing dtype of the sparse matrix into a new dtype.
    :param sparse_mat: input sparse matrix whose dtype is going to be changed
    :param dtype: desired dtype for the output sparse matrix
    :return: sparse_mat_out: result of converting the dtype of sparse_mat
    """
    if selected_splinalg_library == "python-graphblas":
        sparse_mat_out = sparse_mat.dup(dtype=dtype)
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat_out = sparse_mat.astype(dtype)
    else:
        sparse_mat_out = None
    return sparse_mat_out


def to_coo(sparse_mat: Sparse_Matrix, copy: bool = True) -> Sparse_Matrix:
    if selected_splinalg_library == "python-graphblas":
        if copy:
            sparse_mat_out = sparse_mat.dup()
        else:
            sparse_mat_out = sparse_mat
    elif selected_splinalg_library == "scipy.sparse":
        # check if already in coo format and if not make a copy in coo format
        if sparse_mat.format == "coo":
            if copy:
                sparse_mat_out = sparse_mat.copy()
            else:
                sparse_mat_out = sparse_mat
        else:
            sparse_mat_out = sparse_mat.tocoo(copy=copy)
    else:
        sparse_mat_out = None
    return sparse_mat_out


def is_coo_format(sparse_mat: Sparse_Matrix) -> bool:
    if selected_splinalg_library == "scipy.sparse":
        is_coo = isinstance(sparse_mat, scisparse.coo_array)
    elif selected_splinalg_library == "python-graphblas":
        is_coo = True  # no explicit notion of format in python-graphblas, so always return True
    else:
        is_coo = False
    return is_coo


def to_csr(sparse_mat: Sparse_Matrix, copy: bool = True) -> Sparse_Matrix:
    if selected_splinalg_library == "python-graphblas":
        # do nothing, since format isn't an explicit feature of gb.Matrix
        if copy:
            sparse_mat_out = sparse_mat.dup()
        else:
            sparse_mat_out = sparse_mat
    elif selected_splinalg_library == "scipy.sparse":
        if sparse_mat.format == "csr":
            if copy:
                sparse_mat_out = sparse_mat.copy()
            else:
                sparse_mat_out = sparse_mat
        else:
            sparse_mat_out = sparse_mat.tocsr(copy=copy)
    else:
        sparse_mat_out = empty()
    return sparse_mat_out


def to_csc(sparse_mat: Sparse_Matrix, copy: bool = True) -> Sparse_Matrix:
    if selected_splinalg_library == "python-graphblas":
        # do nothing, since format isn't an explicit feature of gb.Matrix
        if copy:
            sparse_mat_out = sparse_mat.dup()
        else:
            sparse_mat_out = sparse_mat
    elif selected_splinalg_library == "scipy.sparse":
        if sparse_mat.format == "csc":
            if copy:
                sparse_mat_out = sparse_mat.copy()
            else:
                sparse_mat_out = sparse_mat
        else:
            sparse_mat_out = sparse_mat.tocsc(copy=copy)
    else:
        sparse_mat_out = empty()
    return sparse_mat_out


def to_array(sparse_mat: Sparse_Matrix) -> np.ndarray:
    if selected_splinalg_library == "python-graphblas":
        dense_mat = sparse_mat.to_dense(fill_value=0)
    elif selected_splinalg_library == "scipy.sparse":
        dense_mat = sparse_mat.toarray()
    else:
        dense_mat = np.ndarray(shape=(0,0))
    return dense_mat


def to_scipy(sparse_mat: Sparse_Matrix) -> scisparse.sparray:
    if selected_splinalg_library == "python-graphblas":
        scipy_mat = gb.io.to_scipy_sparse(sparse_mat, format='coo')
    elif selected_splinalg_library == "scipy.sparse":
        scipy_mat = sparse_mat
    else:
        scipy_mat = scisparse.coo_array(([], ([], [])))
    return scipy_mat


def sum_duplicates(sparse_mat: Sparse_Matrix) -> None:
    if selected_splinalg_library == "python-graphblas":
        pass  # doesn't seem to be a necessary thing for python-graphblas matrices
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat.sum_duplicates()
    else:
        pass
    return None


def eliminate_zeros(sparse_mat: Sparse_Matrix) -> None:
    if selected_splinalg_library == "python-graphblas":
        pass  # doesn't seem to be a necessary thing for python-graphblas matrices
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat.eliminate_zeros()
    else:
        pass
    return None


def copy(sparse_mat: Sparse_Matrix) -> Sparse_Matrix:
    if selected_splinalg_library == "python-graphblas":
        sparse_mat_copy = sparse_mat.dup()
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat_copy = sparse_mat.copy()
    else:
        sparse_mat_copy = None
    return sparse_mat_copy


# Arithmetic/algebraic methods
def add(sparse_mat_A: Sparse_Matrix,
        sparse_mat_B: Sparse_Matrix,
        convert_to_coo: bool = True
        ) \
        -> Sparse_Matrix:
    """
    Add two sparse matrices.
    :param sparse_mat_A: first sparse matrix input
    :param sparse_mat_B: second sparse matrix input
    :param convert_to_coo: boolean indicating whether sparse_mat_out should be automatically converted to COOrdinate
        format, default=True
    :return: sparse_mat_out: sum of sparse_matrix_A and sparse_matrix_B
    Notes:
        * sparse_mat_A and sparse_mat_B are assumed to have compatible shapes and dtypes
        * When python-graphblas is the selected sparse linear algebra library, the storage format of the sparse matrix
            is opaque, so the convert_to_coo parameter is not used
    """
    if selected_splinalg_library == "python-graphblas":
        sparse_mat_out = sparse_mat_A + sparse_mat_B
    elif selected_splinalg_library == "scipy.sparse":
        if sparse_mat_A.format == "csr":
            sparse_mat_Acsx = sparse_mat_A
            if sparse_mat_B.format == "csr":
                sparse_mat_Bcsx = sparse_mat_B
            else:
                sparse_mat_Bcsx = sparse_mat_B.tocsr()
        elif sparse_mat_A.format == "csc":
            sparse_mat_Acsx = sparse_mat_A
            if sparse_mat_B.format == "csc":
                sparse_mat_Bcsx = sparse_mat_B
            else:
                sparse_mat_Bcsx = sparse_mat_B.tocsc()
        else:
            sparse_mat_Acsx = sparse_mat_A.tocsr()
            sparse_mat_Bcsx = sparse_mat_B.tocsr()
        sparse_mat_outcsx = sparse_mat_Acsx + sparse_mat_Bcsx
        if convert_to_coo:
            sparse_mat_out = sparse_mat_outcsx.tocoo()
        else:
            sparse_mat_out = sparse_mat_outcsx
    else:
        sparse_mat_out = None
    return sparse_mat_out


def matmul(sparse_mat_A: Sparse_Matrix,
           sparse_mat_B: Sparse_Matrix,
           convert_to_coo: bool = True
           ) \
        -> Sparse_Matrix:
    """
    Matrix multiply two sparse matrices.
    :param sparse_mat_A: first sparse matrix input
    :param sparse_mat_B: second sparse matrix input
    :param convert_to_coo: boolean indicating whether sparse_mat_out should be automatically converted to COOrdinate
        format, default=True
    :return: sparse_mat_out: matrix product sparse_mat_A @ sparse_mat_B
    Notes:
        * sparse_mat_A and sparse_mat_B are assumed to have compatible shapes and dtypes
        * When python-graphblas is the selected sparse linear algebra library, the storage format of the sparse matrix
            is opaque, so the convert_to_coo parameter is not used
        * When scipy.sparse is the selected sparse linear algebra library and the convert_to_coo parameter is set to
            False, the format of the output sparse_mat_out is CSR (compressed sparse row)
    """
    if selected_splinalg_library == "python-graphblas":
        sparse_mat_out = sparse_mat_A @ sparse_mat_B
    elif selected_splinalg_library == "scipy.sparse":
        # ideal format for matrix arithmetic is csr @ csc, so check if already
        # in csr format and otherwise make a copy in that format
        if sparse_mat_A.format == "csr":
            sparse_mat_Acsr = sparse_mat_A
        else:
            sparse_mat_Acsr = sparse_mat_A.tocsr()
        if sparse_mat_B.format == "csc":
            sparse_mat_Bcsc = sparse_mat_B
        else:
            sparse_mat_Bcsc = sparse_mat_B.tocsc()

        sparse_mat_outcsr = sparse_mat_Acsr @ sparse_mat_Bcsc  # csr @ csc -> csr
        if convert_to_coo:
            sparse_mat_out = sparse_mat_outcsr.tocoo()
        else:
            sparse_mat_out = sparse_mat_outcsr
    else:
        sparse_mat_out = None
    return sparse_mat_out


def ewise_mult(sparse_mat_A: Sparse_Matrix, sparse_mat_B: Sparse_Matrix, convert_to_coo: bool = True) -> Sparse_Matrix:
    if selected_splinalg_library == "python-graphblas":
        sparse_mat_out = sparse_mat_A.ewise_mult(sparse_mat_B)
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat_out = sparse_mat_A.multiply(sparse_mat_B)  # would converting to CSR/CSC be better?
    else:
        sparse_mat_out = empty()
    return sparse_mat_out


def nnz(sparse_mat: Sparse_Matrix) -> int:
    if selected_splinalg_library == "python-graphblas":
        sparse_mat_nnz = sparse_mat.nvals
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat_nnz = sparse_mat.nnz
    else:
        sparse_mat_nnz = 0
    return sparse_mat_nnz


def sparse_sum(sparse_mat: Sparse_Matrix, dimension: Optional[int] = None) \
        -> Union[np.ndarray, float]:
    if selected_splinalg_library == "python-graphblas":
        if dimension is None:
            sparse_mat_sum = sparse_mat.reduce()
        elif dimension == 0:
            v = sparse_mat.reduce_vector()
            col, val = v.to_lists()
            sparse_mat_sum = gb.Matrix.from_coo([0] * len(col), col, val)
        elif dimension == 1:
            sparse_mat_T = sparse_mat.transpose()
            v = sparse_mat_T.reduce_vector()
            row, val = v.to_lists()
            sparse_mat_sum = gb.Matrix.from_coo(row, [0] * len(row), val)
        else:
            raise ValueError("Invalid dimension argument.")
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat_sum = sparse_mat.sum(dimension)
    else:
        sparse_mat_sum = 0
    return sparse_mat_sum


def logical(sparse_mat: Sparse_Matrix, copy: bool = False) -> Sparse_Matrix:
    if selected_splinalg_library == "python-graphblas":
        sparse_mat_logical = sparse_mat.dup() if copy else sparse_mat
        sparse_mat_logical = sparse_mat_logical.apply(gb.unary.one)
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat_logical = sparse_mat.copy() if copy else sparse_mat
        sparse_mat_logical.data[:] = 1.0
    else:
        sparse_mat_logical = empty()
    return sparse_mat_logical


def transpose(sparse_mat: Sparse_Matrix, copy: bool = False) -> Sparse_Matrix:
    if selected_splinalg_library == "python-graphblas":
        sparse_mat_t = sparse_mat.dup() if copy else sparse_mat
        sparse_mat_t = sparse_mat_t.T
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat_t = sparse_mat.copy() if copy else sparse_mat
        sparse_mat_t = sparse_mat_t.transpose()
    else:
        sparse_mat_t = empty()
    return sparse_mat_t


def coo_canonicalize(sparse_mat: Sparse_Matrix, copy: bool = False) -> Sparse_Matrix:
    if selected_splinalg_library == "python-graphblas":
        # no notion of 'canonical' in python-graphblas, so do nothing
        if copy:
            sparse_mat_can = sparse_mat.dup()
        else:
            sparse_mat_can = sparse_mat
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat_can = sparse_mat.tocsr(copy=copy).tocoo()
    else:
        sparse_mat_can = empty()
    return sparse_mat_can


def sparse_equal(sparse_mat_A: Sparse_Matrix, sparse_mat_B: Sparse_Matrix, rtol: float =1e-05, atol: float =1e-08):
    """Test whether two COO sparse matrices are equal."""
    if selected_splinalg_library == "python-graphblas":
        is_equal = sparse_mat_A.isclose(sparse_mat_B, rel_tol=rtol, abs_tol=atol)
    elif selected_splinalg_library == "scipy.sparse":
        # check if dimensions mismatched
        if sparse_mat_A.shape != sparse_mat_B.shape:
            is_equal = False
        else:
            # convert to csr format as needed
            if sparse_mat_A.format == "csr":
                sparse_mat_Acsr = sparse_mat_A
            else:
                sparse_mat_Acsr = sparse_mat_A.tocsr()
            if sparse_mat_B.format == "csr":
                sparse_mat_Bcsr = sparse_mat_B
            else:
                sparse_mat_Bcsr = sparse_mat_B.tocsr()

            diff = np.abs(sparse_mat_Acsr - sparse_mat_Bcsr)
            diff_max = diff.max() if diff.nnz else 0
            sparse_mat_Acsr_abs_max = np.abs(sparse_mat_Acsr).max() if sparse_mat_Acsr.nnz else 0
            sparse_mat_Bcsr_abs_max = np.abs(sparse_mat_Bcsr).max() if sparse_mat_Bcsr.nnz else 0
            tol = atol + rtol * max(sparse_mat_Acsr_abs_max, sparse_mat_Bcsr_abs_max)
            is_equal = (diff_max <= tol)
    else:
        is_equal = False
    return is_equal


# Getter methods
def get_data(sparse_mat: Sparse_Matrix) -> np.ndarray:
    if selected_splinalg_library == "python-graphblas":
        sparse_mat_data = sparse_mat.vals
    elif selected_splinalg_library == "scipy.sparse":
        # check if already in coo format and has canonical format
        if sparse_mat.format == "coo" and sparse_mat.has_canonical_format:
            sparse_mat_coo = sparse_mat
        else:
            sparse_mat_coo = coo_canonicalize(sparse_mat)
        sparse_mat_data = sparse_mat_coo.data
    else:
        sparse_mat_data = np.array([])
    return sparse_mat_data


def get_dtype(sparse_mat: Sparse_Matrix) -> Union[np.dtype, type(float), type(int), None]:
    if selected_splinalg_library == "python-graphblas":
        sparse_mat_dtype = sparse_mat.dtype
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat_dtype = sparse_mat.dtype
    else:
        sparse_mat_dtype = None
    return sparse_mat_dtype


def get_shape(sparse_mat: Sparse_Matrix) -> Tuple[int, int]:
    if selected_splinalg_library == "python-graphblas":
        sparse_mat_shape = sparse_mat.shape
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat_shape = sparse_mat.shape
    else:
        sparse_mat_shape = (0, 0)
    return sparse_mat_shape


def get_coo_params(sparse_mat: Sparse_Matrix) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if selected_splinalg_library == "python-graphblas":
        row, col, data = sparse_mat.to_coo()
    elif selected_splinalg_library == "scipy.sparse":
        # check if already in coo format and if not make a copy in coo format
        sparse_mat_coo = coo_canonicalize(sparse_mat, copy=True)
        data, row, col = sparse_mat_coo.data, sparse_mat_coo.row, sparse_mat_coo.col
    else:
        data, row, col = np.array([]), np.array([]), np.array([])
    return row, col, data


def get_csr_params(sparse_mat: Sparse_Matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if selected_splinalg_library == "python-graphblas":
        indptr, indices, data = sparse_mat.to_csr()
    elif selected_splinalg_library == "scipy.sparse":
        # check if already in csr format and if not make a copy in csr format
        if sparse_mat.format == "csr":
            sparse_mat_csr = sparse_mat
        else:
            sparse_mat_csr = sparse_mat.tocsr()
        data, indices, indptr = sparse_mat_csr.data, sparse_mat_csr.indices, sparse_mat_csr.indptr
    else:
        return np.array([]), np.array([]), np.array([])
    return data, indices, indptr


# Data extraction and manipulation methods
def get_csc_params(sparse_mat: Sparse_Matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if selected_splinalg_library == "python-graphblas":
        indptr, indices, data = sparse_mat.to_csc()
    elif selected_splinalg_library == "scipy.sparse":
        # check if already in csr format and if not make a copy in csr format
        if sparse_mat.format == "csc":
            sparse_mat_csc = sparse_mat
        else:
            sparse_mat_csc = sparse_mat.tocsc()
        data, indices, indptr = sparse_mat_csc.data, sparse_mat_csc.indices, sparse_mat_csc.indptr
    else:
        return np.array([]), np.array([]), np.array([])
    return data, indices, indptr


def getitem_rows(sparse_mat: Sparse_Matrix, sub_row: Union[list, np.array]) -> Sparse_Matrix:
    if selected_splinalg_library == "python-graphblas":
        sparse_mat_sub = sparse_mat[sub_row, :]
    elif selected_splinalg_library == "scipy.sparse":
        if sparse_mat.format == "csr":
            sparse_mat_csr = sparse_mat
        else:
            sparse_mat_csr = sparse_mat.tocsr()
        sparse_mat_sub_csr = sparse_mat_csr[sub_row, :]
        sparse_mat_sub = sparse_mat_sub_csr.tocoo()
    else:
        sparse_mat_sub = empty()
    return sparse_mat_sub


def getitem_cols(sparse_mat: Sparse_Matrix, sub_col: Union[list, np.array]) -> Sparse_Matrix:
    if selected_splinalg_library == "python-graphblas":
        sparse_mat_sub = sparse_mat[:, sub_col]
    elif selected_splinalg_library == "scipy.sparse":
        if sparse_mat.format == "csc":
            sparse_mat_csc = sparse_mat
        else:
            sparse_mat_csc = sparse_mat.tocsc()
        sparse_mat_sub_csc = sparse_mat_csc[:, sub_col]
        sparse_mat_sub = sparse_mat_sub_csc.tocoo()
    else:
        sparse_mat_sub = empty()
    return sparse_mat_sub


def getvalue(sparse_mat: Sparse_Matrix, row_index: int, col_index: int):
    if selected_splinalg_library == "python-graphblas":
        value = sparse_mat.get(row_index, col_index)
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat_csr = sparse_mat.tocsr()
        value = sparse_mat_csr[row_index, col_index]
    else:
        value = 0
    return value


def update_data(sparse_mat: Sparse_Matrix, new_data: np.array) -> Sparse_Matrix:
    """
    Replace existing data of a sparse matrix with new data.
    :param sparse_mat: input sparse matrix whose data is to be replaced
    :param new_data: new data numpy array replacing sparse_mat's initial data numpy array
    :return: sparse_mat_out:
    Notes:
        * sparse_mat is assumed to be in canonical form, so that, e.g., when it is of COO format and its data is
        extracted the values are sorted first by row and then by column
            - Python-GraphBLAS appears to always be in canonical form
            - scipy.sparse cannot be assumed to be in canonical form but does track it with a 'has_canonical_format'
                attribute
        * new_data parameter is assumed to line up perfectly with the extracted data array; e.g., if sparse_mat is
            stored in COO or CSR format then new_data is assumed to be sorted first by row and then by column, while
            if sparse_mat is stored in CSC format then new_data is assumed to be sorted first by column and then by row
        * dtype conversions are not specifically taken into account (though perhaps the selected sparse linear algebra
            library might take care of that itself)
    """
    if selected_splinalg_library == "python-graphblas":
        data, row, col = get_coo_params(sparse_mat)
        assert np.size(data) == np.size(new_data)
        sparse_mat_out = gb.Matrix.from_coo(row, col, values=data)
    elif selected_splinalg_library == "scipy.sparse":
        sparse_mat_format = sparse_mat.format
        sparse_mat_out = coo_canonicalize(sparse_mat)
        assert np.size(sparse_mat_out.data) == np.size(new_data)
        sparse_mat_out.data = new_data
        sparse_mat_out = sparse_mat_out.asformat(sparse_mat_format)
    else:
        sparse_mat_out = empty()
    return sparse_mat_out
