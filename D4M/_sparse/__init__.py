import os
from typing import Optional, Tuple, Union

import numpy as np


supported_splinalg_libraries = {"scipy.sparse", "python-graphblas"}
selected_splinalg_library = os.environ.get("D4M_SPARSE_BACKEND", "scipy.sparse")

if selected_splinalg_library not in supported_splinalg_libraries:
    raise ValueError(
        "D4M_SPARSE_BACKEND must be one of "
        + ", ".join(sorted(supported_splinalg_libraries))
        + "."
    )

if selected_splinalg_library == "scipy.sparse":
    from .scipy_backend import ScipyBackend

    _backend = ScipyBackend()
else:
    try:
        from .graphblas_backend import GraphBLASBackend
    except ModuleNotFoundError as e:
        if e.name != "graphblas":
            raise
        raise ImportError(
            "The python-graphblas package is required when "
            "D4M_SPARSE_BACKEND=python-graphblas."
        ) from e

    _backend = GraphBLASBackend()

Sparse_Matrix = _backend.sparse_matrix_type


def init(sparse_like, *args, **kwargs) -> Sparse_Matrix:
    return _backend.init(sparse_like, *args, **kwargs)


def from_coo(
    row: np.ndarray,
    col: np.ndarray,
    val: np.ndarray,
    shape: Optional[Tuple[int, int]] = None,
    dtype: Optional[Union[np.dtype, type[float], type[int]]] = None,
) -> Sparse_Matrix:
    """Initialize a sparse matrix using COOrdinate data.

    SciPy results are returned in canonical COO format. ``shape`` is inferred
    from the coordinates when omitted.
    """
    return _backend.from_coo(row, col, val, shape=shape, dtype=dtype)


def from_dense(
    np_mat: np.ndarray,
    missing_value: Union[int, float] = 0,
    dtype: Optional[Union[np.dtype, type[float], type[int]]] = None,
) -> Sparse_Matrix:
    return _backend.from_dense(np_mat, missing_value=missing_value, dtype=dtype)


def empty() -> Sparse_Matrix:
    """Return an empty sparse matrix with shape ``(0, 0)``."""
    return _backend.empty()


def astype(
    sparse_mat: Sparse_Matrix, dtype: Union[np.dtype, type(int), type(float)]
) -> Sparse_Matrix:
    """Return ``sparse_mat`` converted to ``dtype``."""
    return _backend.astype(sparse_mat, dtype)


def to_coo(sparse_mat: Sparse_Matrix, copy: bool = True) -> Sparse_Matrix:
    return _backend.to_coo(sparse_mat, copy=copy)


def is_coo_format(sparse_mat: Sparse_Matrix) -> bool:
    return _backend.is_coo_format(sparse_mat)


def to_csr(sparse_mat: Sparse_Matrix, copy: bool = True) -> Sparse_Matrix:
    return _backend.to_csr(sparse_mat, copy=copy)


def to_csc(sparse_mat: Sparse_Matrix, copy: bool = True) -> Sparse_Matrix:
    return _backend.to_csc(sparse_mat, copy=copy)


def to_array(sparse_mat: Sparse_Matrix) -> np.ndarray:
    return _backend.to_array(sparse_mat)


def to_scipy(sparse_mat: Sparse_Matrix):
    return _backend.to_scipy(sparse_mat)


def sum_duplicates(sparse_mat: Sparse_Matrix) -> None:
    return _backend.sum_duplicates(sparse_mat)


def eliminate_zeros(sparse_mat: Sparse_Matrix) -> None:
    return _backend.eliminate_zeros(sparse_mat)


def copy(sparse_mat: Sparse_Matrix) -> Sparse_Matrix:
    return _backend.copy(sparse_mat)


def add(
    sparse_mat_A: Sparse_Matrix,
    sparse_mat_B: Sparse_Matrix,
    convert_to_coo: bool = True,
) -> Sparse_Matrix:
    """Add two sparse matrices.

    ``convert_to_coo`` controls the SciPy result format and has no storage
    format effect for GraphBLAS.
    """
    return _backend.add(sparse_mat_A, sparse_mat_B, convert_to_coo=convert_to_coo)


def matmul(
    sparse_mat_A: Sparse_Matrix,
    sparse_mat_B: Sparse_Matrix,
    convert_to_coo: bool = True,
) -> Sparse_Matrix:
    """Matrix multiply two sparse matrices.

    With SciPy, the result is COO by default and CSR when
    ``convert_to_coo=False``. GraphBLAS has no exposed storage format.
    """
    return _backend.matmul(
        sparse_mat_A, sparse_mat_B, convert_to_coo=convert_to_coo
    )


def ewise_mult(
    sparse_mat_A: Sparse_Matrix,
    sparse_mat_B: Sparse_Matrix,
    convert_to_coo: bool = True,
) -> Sparse_Matrix:
    return _backend.ewise_mult(
        sparse_mat_A, sparse_mat_B, convert_to_coo=convert_to_coo
    )


def nnz(sparse_mat: Sparse_Matrix) -> int:
    return _backend.nnz(sparse_mat)


def sparse_sum(
    sparse_mat: Sparse_Matrix, dimension: Optional[int] = None
) -> Union[np.ndarray, float]:
    return _backend.sparse_sum(sparse_mat, dimension=dimension)


def logical(sparse_mat: Sparse_Matrix, copy: bool = False) -> Sparse_Matrix:
    return _backend.logical(sparse_mat, copy=copy)


def transpose(sparse_mat: Sparse_Matrix, copy: bool = False) -> Sparse_Matrix:
    return _backend.transpose(sparse_mat, copy=copy)


def coo_canonicalize(
    sparse_mat: Sparse_Matrix, copy: bool = False
) -> Sparse_Matrix:
    return _backend.coo_canonicalize(sparse_mat, copy=copy)


def sparse_equal(
    sparse_mat_A: Sparse_Matrix,
    sparse_mat_B: Sparse_Matrix,
    rtol: float = 1e-05,
    atol: float = 1e-08,
):
    """Test whether two sparse matrices are equal within a tolerance."""
    return _backend.sparse_equal(sparse_mat_A, sparse_mat_B, rtol=rtol, atol=atol)


def get_data(sparse_mat: Sparse_Matrix) -> np.ndarray:
    return _backend.get_data(sparse_mat)


def get_dtype(
    sparse_mat: Sparse_Matrix,
) -> Union[np.dtype, type(float), type(int), None]:
    return _backend.get_dtype(sparse_mat)


def get_shape(sparse_mat: Sparse_Matrix) -> Tuple[int, int]:
    return _backend.get_shape(sparse_mat)


def get_coo_params(
    sparse_mat: Sparse_Matrix,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _backend.get_coo_params(sparse_mat)


def get_csr_params(
    sparse_mat: Sparse_Matrix,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _backend.get_csr_params(sparse_mat)


def get_csc_params(
    sparse_mat: Sparse_Matrix,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _backend.get_csc_params(sparse_mat)


def getitem_rows(
    sparse_mat: Sparse_Matrix, sub_row: Union[list, np.array]
) -> Sparse_Matrix:
    return _backend.getitem_rows(sparse_mat, sub_row)


def getitem_cols(
    sparse_mat: Sparse_Matrix, sub_col: Union[list, np.array]
) -> Sparse_Matrix:
    return _backend.getitem_cols(sparse_mat, sub_col)


def getvalue(sparse_mat: Sparse_Matrix, row_index: int, col_index: int):
    return _backend.getvalue(sparse_mat, row_index, col_index)


def update_data(sparse_mat: Sparse_Matrix, new_data: np.array) -> Sparse_Matrix:
    """Return a matrix whose existing stored entries contain ``new_data``.

    The input is assumed to be canonical, and ``new_data`` must align with its
    backend-specific canonical data ordering.
    """
    return _backend.update_data(sparse_mat, new_data)
