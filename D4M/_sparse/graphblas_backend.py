import numpy as np
import graphblas as gb


class GraphBLASBackend:
    sparse_matrix_type = gb.Matrix

    def init(self, sparse_like, *args, **kwargs):
        try:
            return gb.Matrix(sparse_like, *args, **kwargs)
        except ValueError as e:
            raise e

    def from_coo(self, row, col, val, shape=None, dtype=None):
        shape = (None, None) if shape is None else shape
        return gb.Matrix.from_coo(
            row, col, val, nrows=shape[0], ncols=shape[1], dtype=dtype
        )

    def from_dense(self, np_mat, missing_value=0, dtype=None):
        return gb.Matrix.from_dense(np_mat, missing_value=missing_value, dtype=dtype)

    def empty(self):
        return gb.Matrix(gb.dtypes.FP64, nrows=0, ncols=0)

    def astype(self, sparse_mat, dtype):
        return sparse_mat.dup(dtype=dtype)

    def to_coo(self, sparse_mat, copy=True):
        return sparse_mat.dup() if copy else sparse_mat

    def is_coo_format(self, sparse_mat):
        return True

    def to_csr(self, sparse_mat, copy=True):
        return sparse_mat.dup() if copy else sparse_mat

    def to_csc(self, sparse_mat, copy=True):
        return sparse_mat.dup() if copy else sparse_mat

    def to_array(self, sparse_mat):
        return sparse_mat.to_dense(fill_value=0)

    def to_scipy(self, sparse_mat):
        return gb.io.to_scipy_sparse(sparse_mat, format="coo")

    def sum_duplicates(self, sparse_mat):
        return None

    def eliminate_zeros(self, sparse_mat):
        return None

    def copy(self, sparse_mat):
        return sparse_mat.dup()

    def add(self, sparse_mat_A, sparse_mat_B, convert_to_coo=True):
        return sparse_mat_A + sparse_mat_B

    def matmul(self, sparse_mat_A, sparse_mat_B, convert_to_coo=True):
        return sparse_mat_A @ sparse_mat_B

    def ewise_mult(self, sparse_mat_A, sparse_mat_B, convert_to_coo=True):
        return sparse_mat_A.ewise_mult(sparse_mat_B)

    def nnz(self, sparse_mat):
        return sparse_mat.nvals

    def sparse_sum(self, sparse_mat, dimension=None):
        if dimension is None:
            return sparse_mat.reduce()
        if dimension == 0:
            v = sparse_mat.reduce_vector()
            col, val = v.to_lists()
            return gb.Matrix.from_coo([0] * len(col), col, val)
        if dimension == 1:
            sparse_mat_T = sparse_mat.transpose()
            v = sparse_mat_T.reduce_vector()
            row, val = v.to_lists()
            return gb.Matrix.from_coo(row, [0] * len(row), val)
        raise ValueError("Invalid dimension argument.")

    def logical(self, sparse_mat, copy=False):
        sparse_mat_logical = sparse_mat.dup() if copy else sparse_mat
        return sparse_mat_logical.apply(gb.unary.one)

    def transpose(self, sparse_mat, copy=False):
        sparse_mat_t = sparse_mat.dup() if copy else sparse_mat
        return sparse_mat_t.T

    def coo_canonicalize(self, sparse_mat, copy=False):
        return sparse_mat.dup() if copy else sparse_mat

    def sparse_equal(self, sparse_mat_A, sparse_mat_B, rtol=1e-05, atol=1e-08):
        return sparse_mat_A.isclose(sparse_mat_B, rel_tol=rtol, abs_tol=atol)

    def get_data(self, sparse_mat):
        _, _, sparse_mat_data = sparse_mat.to_coo()
        return sparse_mat_data

    def get_dtype(self, sparse_mat):
        return sparse_mat.dtype

    def get_shape(self, sparse_mat):
        return sparse_mat.shape

    def get_coo_params(self, sparse_mat):
        return sparse_mat.to_coo()

    def get_csr_params(self, sparse_mat):
        indptr, indices, data = sparse_mat.to_csr()
        return data, indices, indptr

    def get_csc_params(self, sparse_mat):
        indptr, indices, data = sparse_mat.to_csc()
        return data, indices, indptr

    def getitem_rows(self, sparse_mat, sub_row):
        sub_row = np.asarray(sub_row)
        if sub_row.dtype == bool:
            sub_row = np.flatnonzero(sub_row)
        return sparse_mat[sub_row, :].new()

    def getitem_cols(self, sparse_mat, sub_col):
        sub_col = np.asarray(sub_col)
        if sub_col.dtype == bool:
            sub_col = np.flatnonzero(sub_col)
        return sparse_mat[:, sub_col].new()

    def getvalue(self, sparse_mat, row_index, col_index):
        return sparse_mat.get(row_index, col_index)

    def update_data(self, sparse_mat, new_data):
        data, row, col = self.get_coo_params(sparse_mat)
        assert np.size(data) == np.size(new_data)
        return gb.Matrix.from_coo(row, col, values=data)
