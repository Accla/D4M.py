from scipy import sparse as scisparse
try:
    import pygraphblas as gb
except ImportError:
    gb = None
    _graphblas = False
    sparse_mat = scisparse.spmatrix
else:
    _graphblas = True
    sparse_mat = gb.Matrix

import numpy as np
from typing import Union, Tuple, Optional


def sparse_init(row: np.ndarray, col: np.ndarray, val: np.ndarray, shape=None) -> sparse_mat:
    if _graphblas:
        A = gb.Matrix.from_lists(row, col, val)
        A.shape = shape
        return A
    else:
        return scisparse.coo_matrix((val, (row, col)), shape=shape)


def sparse_add(A: sparse_mat, B: sparse_mat) -> sparse_mat:
    if _graphblas:
        return A + B
    else:
        Acsr, Bcsr = A.tocsr(), B.tocsr()
        Ccsr = Acsr + Bcsr
        return Ccsr.tocoo()


def sparse_matmul(A: sparse_mat, B: sparse_mat) -> sparse_mat:
    if _graphblas:
        return A @ B
    else:
        Acsr, Bcsr = A.tocsr(), B.tocsr()
        Ccsr = Acsr @ Bcsr
        return Ccsr.tocoo()


def sparse_size(A: sparse_mat) -> Tuple[float, float]:
    if _graphblas:
        return A.nrows, A.ncols
    else:
        return A.size


def sparse_sum(A: sparse_mat, dimension: Optional[int] = None) -> Union[np.ndarray, float]:
    if _graphblas:
        if dimension is None:
            return A.reduce()
        elif dimension == 0:
            v = A.reduce_vector()
            col, val = v.to_lists()
            return gb.Matrix.from_lists([0] * len(col), col, val)
        elif dimension == 1:
            At = A.transpose()
            v = At.reduce_vector()
            row, val = v.to_lists()
            return gb.Matrix.from_lists(row, [0] * len(row), val)
        else:
            raise ValueError("Invalid dimension argument.")
    else:
        return A.sum(dimension)


def sparse_empty() -> sparse_mat:
    if _graphblas:
        return gb.Matrix.sparse(gb.types.FP64, nrows=0, ncols=0)
    else:
        return scisparse.coo_matrix(([], ([], [])), shape=(0, 0))


def sparse_get_coo_params(A: sparse_mat) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _graphblas:
        data, row, col = A.rows, A.cols, A.vals
        return np.array(data), np.array(row), np.array(col)
    else:
        A_coo = A.tocoo()
        return A_coo.data, A_coo.row, A_coo.col


def sparse_get_csr_params(A: sparse_mat) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _graphblas:
        A_csr = A.to_scipy_sparse(format="csr")
    else:
        A_csr = A.tocsr()
    return A_csr.data, A_csr.indices, A_csr.indptr
