import numpy as np
import scipy.sparse as scisparse


class ScipyBackend:
    sparse_matrix_type = (
        scisparse.sparray,
        scisparse.spmatrix,
    )

    def init(self, sparse_like, *args, **kwargs):
        try:
            return scisparse.coo_array(sparse_like, *args, **kwargs)
        except ValueError as e:
            raise e

    def from_coo(self, row, col, val, shape=None, dtype=None):
        sparse_mat = scisparse.coo_array((val, (row, col)), shape=shape, dtype=dtype)
        return sparse_mat.tocsr().tocoo()

    def from_dense(self, np_mat, missing_value=0, dtype=None):
        sparse_mat_out = scisparse.coo_array(np_mat, dtype=dtype)
        if missing_value == 0:
            sparse_mat_out.eliminate_zeros()
        else:
            sparse_mat_out.data += -missing_value
            sparse_mat_out.eliminate_zeros()
            sparse_mat_out.data += missing_value
        return sparse_mat_out

    def empty(self):
        return scisparse.coo_matrix(([], ([], [])), shape=(0, 0))

    def astype(self, sparse_mat, dtype):
        return sparse_mat.astype(dtype)

    def to_coo(self, sparse_mat, copy=True):
        if sparse_mat.format == "coo":
            return sparse_mat.copy() if copy else sparse_mat
        return sparse_mat.tocoo(copy=copy)

    def is_coo_format(self, sparse_mat):
        return isinstance(sparse_mat, scisparse.coo_array)

    def to_csr(self, sparse_mat, copy=True):
        if sparse_mat.format == "csr":
            return sparse_mat.copy() if copy else sparse_mat
        return sparse_mat.tocsr(copy=copy)

    def to_csc(self, sparse_mat, copy=True):
        if sparse_mat.format == "csc":
            return sparse_mat.copy() if copy else sparse_mat
        return sparse_mat.tocsc(copy=copy)

    def to_array(self, sparse_mat):
        return sparse_mat.toarray()

    def to_scipy(self, sparse_mat):
        return sparse_mat

    def sum_duplicates(self, sparse_mat):
        sparse_mat.sum_duplicates()
        return None

    def eliminate_zeros(self, sparse_mat):
        sparse_mat.eliminate_zeros()
        return None

    def copy(self, sparse_mat):
        return sparse_mat.copy()

    def add(self, sparse_mat_A, sparse_mat_B, convert_to_coo=True):
        if sparse_mat_A.format == "csr":
            sparse_mat_Acsx = sparse_mat_A
            sparse_mat_Bcsx = (
                sparse_mat_B
                if sparse_mat_B.format == "csr"
                else sparse_mat_B.tocsr()
            )
        elif sparse_mat_A.format == "csc":
            sparse_mat_Acsx = sparse_mat_A
            sparse_mat_Bcsx = (
                sparse_mat_B
                if sparse_mat_B.format == "csc"
                else sparse_mat_B.tocsc()
            )
        else:
            sparse_mat_Acsx = sparse_mat_A.tocsr()
            sparse_mat_Bcsx = sparse_mat_B.tocsr()
        sparse_mat_outcsx = sparse_mat_Acsx + sparse_mat_Bcsx
        return sparse_mat_outcsx.tocoo() if convert_to_coo else sparse_mat_outcsx

    def matmul(self, sparse_mat_A, sparse_mat_B, convert_to_coo=True):
        sparse_mat_Acsr = (
            sparse_mat_A if sparse_mat_A.format == "csr" else sparse_mat_A.tocsr()
        )
        sparse_mat_Bcsc = (
            sparse_mat_B if sparse_mat_B.format == "csc" else sparse_mat_B.tocsc()
        )
        sparse_mat_outcsr = sparse_mat_Acsr @ sparse_mat_Bcsc
        return sparse_mat_outcsr.tocoo() if convert_to_coo else sparse_mat_outcsr

    def ewise_mult(self, sparse_mat_A, sparse_mat_B, convert_to_coo=True):
        return sparse_mat_A.multiply(sparse_mat_B)

    def nnz(self, sparse_mat):
        return sparse_mat.nnz

    def sparse_sum(self, sparse_mat, dimension=None):
        return sparse_mat.sum(dimension)

    def logical(self, sparse_mat, copy=False):
        sparse_mat_logical = sparse_mat.copy() if copy else sparse_mat
        sparse_mat_logical.data[:] = 1.0
        return sparse_mat_logical

    def transpose(self, sparse_mat, copy=False):
        sparse_mat_t = sparse_mat.copy() if copy else sparse_mat
        return sparse_mat_t.transpose()

    def coo_canonicalize(self, sparse_mat, copy=False):
        return sparse_mat.tocsr(copy=copy).tocoo()

    def sparse_equal(self, sparse_mat_A, sparse_mat_B, rtol=1e-05, atol=1e-08):
        if sparse_mat_A.shape != sparse_mat_B.shape:
            return False
        sparse_mat_Acsr = (
            sparse_mat_A if sparse_mat_A.format == "csr" else sparse_mat_A.tocsr()
        )
        sparse_mat_Bcsr = (
            sparse_mat_B if sparse_mat_B.format == "csr" else sparse_mat_B.tocsr()
        )
        diff = np.abs(sparse_mat_Acsr - sparse_mat_Bcsr)
        diff_max = diff.max() if diff.nnz else 0
        sparse_mat_Acsr_abs_max = (
            np.abs(sparse_mat_Acsr).max() if sparse_mat_Acsr.nnz else 0
        )
        sparse_mat_Bcsr_abs_max = (
            np.abs(sparse_mat_Bcsr).max() if sparse_mat_Bcsr.nnz else 0
        )
        tol = atol + rtol * max(sparse_mat_Acsr_abs_max, sparse_mat_Bcsr_abs_max)
        return diff_max <= tol

    def get_data(self, sparse_mat):
        if sparse_mat.format == "coo" and sparse_mat.has_canonical_format:
            sparse_mat_coo = sparse_mat
        else:
            sparse_mat_coo = self.coo_canonicalize(sparse_mat)
        return sparse_mat_coo.data

    def get_dtype(self, sparse_mat):
        return sparse_mat.dtype

    def get_shape(self, sparse_mat):
        return sparse_mat.shape

    def get_coo_params(self, sparse_mat):
        sparse_mat_coo = self.coo_canonicalize(sparse_mat, copy=True)
        return sparse_mat_coo.row, sparse_mat_coo.col, sparse_mat_coo.data

    def get_csr_params(self, sparse_mat):
        sparse_mat_csr = (
            sparse_mat if sparse_mat.format == "csr" else sparse_mat.tocsr()
        )
        return sparse_mat_csr.data, sparse_mat_csr.indices, sparse_mat_csr.indptr

    def get_csc_params(self, sparse_mat):
        sparse_mat_csc = (
            sparse_mat if sparse_mat.format == "csc" else sparse_mat.tocsc()
        )
        return sparse_mat_csc.data, sparse_mat_csc.indices, sparse_mat_csc.indptr

    def getitem_rows(self, sparse_mat, sub_row):
        sparse_mat_csr = (
            sparse_mat if sparse_mat.format == "csr" else sparse_mat.tocsr()
        )
        return sparse_mat_csr[sub_row, :].tocoo()

    def getitem_cols(self, sparse_mat, sub_col):
        sparse_mat_csc = (
            sparse_mat if sparse_mat.format == "csc" else sparse_mat.tocsc()
        )
        return sparse_mat_csc[:, sub_col].tocoo()

    def getvalue(self, sparse_mat, row_index, col_index):
        return sparse_mat.tocsr()[row_index, col_index]

    def update_data(self, sparse_mat, new_data):
        sparse_mat_format = sparse_mat.format
        sparse_mat_out = self.coo_canonicalize(sparse_mat)
        assert np.size(sparse_mat_out.data) == np.size(new_data)
        sparse_mat_out.data = new_data
        return sparse_mat_out.asformat(sparse_mat_format)
