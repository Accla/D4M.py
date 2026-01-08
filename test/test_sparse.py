import pytest
import numpy as np
from numbers import Number
import scipy.sparse as scisparse
import graphblas as gb

import D4M.assoc
import D4M.util
import D4M._sparse
from D4M.util import _replace_default_args


@pytest.mark.parametrize(
    "test_row,test_col,test_val,test_shape,test_dtype,scisparse_sparse_mat,gb_sparse_mat",
    [
        (
            [0, 1, 2],
            [0, 1, 2],
            [1, 1, 1],
            None,
            None,
            scisparse.coo_array(([1, 1, 1], ([0, 1, 2], [0, 1, 2]))),
            gb.Matrix.from_coo([0, 1, 2], [0, 1, 2], [1, 1, 1])
        ),
        (
            [0, 1, 2],
            [0, 1, 2],
            [1, 1, 1],
            (4, 4),
            int,
            scisparse.coo_array(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(4, 4), dtype=int),
            gb.Matrix.from_coo([0, 1, 2], [0, 1, 2], [1, 1, 1], nrows=4, ncols=4, dtype=int)
        ),
        (
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            np.array([1, 1, 1]),
            None,
            None,
            scisparse.coo_array(([1, 1, 1], ([0, 1, 2], [0, 1, 2]))),
            gb.Matrix.from_coo([0, 1, 2], [0, 1, 2], [1, 1, 1])
        ),
        (
            [0, 1, 2],
            [0, 1, 2],
            [1, 1, 1],
            None,
            float,
            scisparse.coo_array(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), dtype=float),
            gb.Matrix.from_coo([0, 1, 2], [0, 1, 2], [1, 1, 1], dtype=float)
        )
    ]
)
def test_from_coo(test_row, test_col, test_val, test_shape, test_dtype, scisparse_sparse_mat, gb_sparse_mat):
    if D4M._sparse.selected_splinalg_library == "scipy.sparse":
        exp_sparse_mat = scisparse.coo_array((test_val, (test_row, test_col)), shape=test_shape, dtype=test_dtype)
        sparse_mat = scisparse_sparse_mat
    elif D4M._sparse.selected_splinalg_library == "python-graphblas":
        exp_sparse_mat = gb.Matrix.from_coo(test_row,
                                            test_col,
                                            test_val,
                                            nrows=test_shape[0],
                                            ncols=test_shape[1],
                                            dtype=test_dtype
                                            )
        sparse_mat = gb_sparse_mat
    else:
        exp_sparse_mat = D4M._sparse.empty()
        sparse_mat = D4M._sparse.empty()
    assert D4M._sparse.sparse_equal(exp_sparse_mat, sparse_mat)
