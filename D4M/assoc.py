# Import packages
from scipy import io, sparse
import numpy as np
import matplotlib.pyplot as plt
import csv
import shutil
import warnings
import copy as cpy
from numbers import Number
from typing import Union, Tuple, Optional, Callable, Sequence, List, Dict, Any


# Use List & Dict for backwards (<3.9) compatibility

import D4M.util as util
import D4M._sparse as _sparse

operation_dict = util.operation_dict()

KeyVal = Union[str, Number]
StrList = Union[str, Sequence[str]]
ArrayLike = Union[KeyVal, Sequence[KeyVal], np.ndarray]
Selectable = Union[ArrayLike, slice, Callable]


# Main class and methods

# noinspection PyPep8Naming
class Assoc:
    """Associative arrays, supporting basic sparse linear algebra on sparse matrices with
        values of variable (string or numerical) type, variable operations (plus-times, max-min, etc)
        and row and column indices of variable (string or numerical) type.
    Structure:
        row = sorted array of strings/numbers (row indices)
        col = sorted array of strings/numbers (column indices)
        val = sorted array of values
            or 1.0 to indicate that all the values are numerical and stored in adj
        adj = adjacency array implemented as sparse matrix (COO format)
            if val==1.0 then adj is a sparse matrix containing the actual numerical values
            otherwise
            adj[row_id,col_id] = (index of corresponding value in val) + 1, or 0 if empty
    Note: Associative arrays are assumed to contain no explicit null values ('', 0, None).
    """

    null_values = {"", 0, None}

    def __init__(
        self,
        row: ArrayLike,
        col: ArrayLike,
        val: ArrayLike,
        adj: Optional[_sparse.Sparse_Matrix] = None,
        aggregate: Union[Callable[[KeyVal, KeyVal], KeyVal], str] = min,
        prevent_upcasting: bool = False,
        convert_val: bool = False,
    ):
        """Construct an associative array either from an existing sparse matrix (e.g., gb.Matrix or
        scipy.sparse.coo_matrix) or from row, column, and value triples.
            Usage:
                A = Assoc(row,col,val)
                A = Assoc(row,col,val,aggregate=func)
                A = Assoc(row,col,number,aggregate=func)
                A = Assoc(row,col,val,adj=sparse_matrix)
            Inputs:
                row, col, val = each either:
                    - a string of (delimiter separated) values/keys (last character is taken as the delimiter), or
                    - a sequence of string or numerical values/keys
                adj = (Optional, default is None) a sparse matrix to be used as the adjacency array, where:
                    - if val == 1.0, then the entries of adj are the values of the associative array, with row and
                        column indices with respect to unique sorted entries in row and col, resp.
                    - otherwise, then the entries of adj are the (1-indexed) indices with respect to unique sorted
                        entries in val, with row and column indices with respect to unique sorted entries in row and
                        col, resp.
                aggregate = (Optional, default is min) aggregate function to handle (row_key, col_key) collisions,
                    either:
                        - two-input callable compatible with supplied values,
                        - string representing a supported function (e.g. 'add', 'first', 'last', 'min', 'max')
                        - 'unique', indicating no collisions; when adj is supplied, indicates that row, col, val
                            are already sorted and unique
                prevent_upcasting = (Optional, not fully implemented, default is False) Boolean indicating if
                    row/col/val entries should keep their types (e.g., [1, 2.5] won't be upcast to [1.0, 2.5])
                    [at the cost of potential loss of performance]
                convert = (Optional, not fully implemented, default is False) Boolean indicating if values should be
                    converted to numerical data when possible
            Outputs:
                A = Associative array made from the triples row, col, and val
            Examples:
                A = Assoc('r1,r2,', 'c1,c2,', 'v1;v2;')
                A = Assoc(['r1','r2'], 'c1/c2/', [1,2], 'add')
                A = Assoc('1,', 'c1,', np.array([3]))
                A = Assoc('r1,r2,', 'c1,c2,', 1.0, [sparse_matrix])
            Notes:
                - If both adj and aggregate are supplied, if aggregate is not 'unique', then its value is ignored.
                - If adj is supplied, it must have shape compatible with the number of unique entries in row and col.
                - To determine whether data is numerical or not, val is sorted. If the last entry is numerical, then
                    all entries are assumed numerical. (Last entry is chosen as Python previously sorted mixed data
                    types by putting numerical data types before non-numerical data types. Currently, mixed data types
                    are not directly supported.)
        """
        if aggregate in operation_dict.keys():
            aggregate = operation_dict[aggregate]

        # Sanitize
        row = util.sanitize(row, prevent_upcasting=prevent_upcasting)
        col = util.sanitize(col, prevent_upcasting=prevent_upcasting)

        row_size, col_size = len(row), len(col)

        if row_size == 0 or col_size == 0 or np.size(val) == 0:
            # Short-circuit if empty assoc
            self.row = np.empty(0)
            self.col = np.empty(0)
            self.val = 1.0  # Considered numerical
            self.adj = _sparse.empty()
        elif adj is not None and aggregate == "unique":
            # Assume everything is already done, except possible sanitization of val
            self.row = row
            self.col = col
            if not isinstance(val, float) or val != 1.0:
                val = util.sanitize(val)
                if convert_val:
                    val = util.str_to_num(val)
            self.val = val
            self.adj = _sparse.to_coo(adj)
        else:
            if adj is not None:
                adj = _sparse.to_coo(adj)
                _sparse.eliminate_zeros(adj)
                adj_row, adj_col, adj_data = _sparse.get_coo_params(adj)

                if isinstance(val, float) and val == 1.0:
                    is_float = True
                    _sparse.sum_duplicates(adj)
                    val = adj_data
                else:
                    is_float = False
                    val = util.sanitize(val)
                    if convert_val:
                        val = util.str_to_num(val)

                (row_dim, col_dim) = _sparse.get_shape(adj)

                unique_row, unique_col, unique_val = (
                    np.unique(row),
                    np.unique(col),
                    np.unique(val),
                )

                error_message = "Invalid input:"
                good_params = [
                    np.size(unique_row) >= row_dim,
                    np.size(unique_col) >= col_dim,
                    np.size(unique_val) >= np.size(np.unique(adj_data)),
                ]
                param_type = ["row indices", "col indices", "values"]
                for index in range(3):
                    if (
                        index > 0
                        and False in good_params[0:index]
                        and not good_params[index]
                    ):
                        error_message += ","
                    if not good_params[index]:
                        error_message += " not enough unique " + param_type[index]
                error_message += "."
                if False in good_params:
                    raise ValueError(error_message)

                new_row, new_col, new_val = unique_row[adj_row], unique_col[adj_col], 0

                if is_float:
                    new_val = adj_data
                else:
                    try:
                        new_val = unique_val[
                            adj_data - np.ones(np.size(adj_data), dtype=int)
                        ]
                    except (TypeError, IndexError):
                        print(
                            "Values in sparse matrix must correspond to elements of val (after sorting and removing "
                            "duplicates)"
                        )

                row, col, val = new_row, new_col, new_val
                row_size, col_size = len(row), len(col)
                aggregate = min
            else:
                pass

            val = util.sanitize(val, prevent_upcasting=prevent_upcasting)
            if convert_val:
                val = util.str_to_num(val)
            val_size = np.size(val)
            max_size = max([row_size, col_size, val_size])
            if row_size == 1:
                row = np.full(max_size, row[0])
                row_size = max_size
            if col_size == 1:
                col = np.full(max_size, col[0])
                col_size = max_size
            if val_size == 1:
                val = np.full(max_size, val[0])
                val_size = max_size

            if min([row_size, col_size, val_size]) < max_size:
                raise ValueError(
                    "Invalid input: row, col, val must have compatible lengths."
                )

            row, col, val = util.aggregate_triples(row, col, val, aggregate)

            null_indices = [
                index
                for index in range(np.size(val))
                if val[index] in Assoc.null_values
            ]
            row, col, val = (
                np.delete(row, null_indices),
                np.delete(col, null_indices),
                np.delete(val, null_indices),
            )

            # Array possibly empty after deletion of null values
            if row_size == 0 or col_size == 0 or np.size(val) == 0:
                self.row = np.empty(0)
                self.col = np.empty(0)
                self.val = 1.0  # Considered numerical
                self.adj = _sparse.empty()
            else:
                # Get unique sorted row and column indices
                self.row, from_row = np.unique(row, return_inverse=True)
                self.col, from_col = np.unique(col, return_inverse=True)
                self.val, from_val = np.unique(val, return_inverse=True)

                # Check if numerical; numpy sorts numerical values to front, so only check last entry
                assert isinstance(self.val, np.ndarray)
                if util.is_numeric(self.val[-1]):
                    if prevent_upcasting:
                        self.adj = _sparse.from_coo(from_row, from_col, val,
                                                    shape=(np.size(self.row), np.size(self.col)))
                    else:
                        self.adj = _sparse.from_coo(from_row, from_col, val,
                                                    shape=(np.size(self.row), np.size(self.col)),
                                                    dtype=float
                                                    )
                    self.val = 1.0
                else:
                    # If not numerical, self.adj has entries given by indices+1 of self.val
                    val_indices = from_val + np.ones(np.size(from_val))
                    self.adj = _sparse.from_coo(from_row, from_col, val_indices, dtype=int)

    def is_canonical(self) -> bool:
        """Determine if self is in canonical form. I.e.:
            - No stored null entries.
            - Every row (resp., col) key corresponds to a nonempty row (resp., column) in self.adj.
            - self.row and self.col are sorted np.ndarrays and contain no duplicate values.
            - self.adj is a sparse matrix in coo form with appropriate shape and dtype.
            - self.val is either an np.ndarray or 1.0.
        Moreover, if self is non-numerical, then additionally:
            - Every value in self.val corresponds to an entry in self.adj.
            - self.val is sorted and contains no duplicate values.
            - Every datum in self.adj's data corresponds to an entry in self.val (i.e., self.adj's data only contains
                elements of {1, 2, 3,..., len(self.val)}.
        """
        canonical = True

        if not (isinstance(self.val, np.ndarray) or self.val == 1.0):
            print("* self.val is not of valid type (np.ndarray or float).")
            canonical = False
        if not (isinstance(self.row, np.ndarray) and isinstance(self.col, np.ndarray)):
            print("* self.row and/or self.col are not of valid type (np.ndarray).")
            canonical = False
        if not isinstance(self.adj, _sparse.Sparse_Matrix):
            print("* self.adj is not of valid type.")
            canonical = False

        # Check if self.row and self.col are sorted
        if not util.np_sorted(self.row) or not util.np_sorted(self.col):
            print("* self.row and/or self.col are not properly sorted.")
            canonical = False

        # Check if self.row or self.col have duplicate elements
        row_set, col_set = set(self.row), set(self.col)
        row_num, col_num = len(row_set), len(col_set)
        if row_num != len(self.row) or col_num != len(self.col):
            print("* self.row and/or self.col contain duplicate elements.")
            canonical = False

        # Check if self.adj has appropriate shape
        if (row_num, col_num) != _sparse.get_shape(self.adj):
            print("* self.adj does not have shape matching self.row and self.col.")
            canonical = False
        if isinstance(self.adj, _sparse.Sparse_Matrix):
            adj_row, adj_col, adj_data = _sparse.get_coo_params(self.adj)
            adj_row_set = set(adj_row)
            adj_col_set = set(adj_col)
            if row_num != len(adj_row_set) or col_num != len(adj_col_set):
                print("* Empty rows or columns present in self.adj.")
                canonical = False

        # check if self.adj is appropriate format (COO when possible)
        if not _sparse.is_coo_format(self.adj):
            print("* Sparse adjacency array is not in proper sparse format.")
            canonical = False

        # Check values for null values
        if isinstance(self.adj, _sparse.Sparse_Matrix) and isinstance(self.val, float):
            if np.isin(0, adj_data):
                print("* Explicit zeros stored in self.adj.")
                canonical = False
        else:
            assert isinstance(self.val, np.ndarray)
            for null_value in Assoc.null_values:
                if np.any(np.isin(null_value, self.val)):
                    print(
                        "* Explicit null values stored in self.val ("
                        + str(null_value)
                        + ")."
                    )
                    canonical = False

            # Check if self.val is sorted
            if not util.np_sorted(self.val):
                print("* self.val is not properly sorted.")
                canonical = False

            # Check if self.val has duplicate elements
            val_set = set(self.val)
            val_num = len(val_set)
            if val_num != len(self.val):
                print("* self.val contains duplicate elements.")
                canonical = False

            # Check if self.adj contains appropriate data, i.e., integers between 1 and val_num

            adj_min = adj_data.min()
            adj_max = adj_data.max()
            if _sparse.get_dtype(self.adj) != int or adj_min < 1 or adj_max > val_num:
                print(
                    "dtype="
                    + str(_sparse.get_dtype(self.adj))
                    + "; min="
                    + str(adj_min)
                    + "; max="
                    + str(adj_max)
                )
                print(
                    "* Values in self.adj are not all (1-indexed) indices of elements of self.val."
                )
                canonical = False

            # Check that every element of self.val arises
            adj_val_set = set(adj_data)
            if val_num != len(adj_val_set):
                print(
                    "* Elements present in self.val which are not reflected in self.adj."
                )
                canonical = False

        return canonical

    def dropzeros(
        self, copy: bool = False
    ) -> "Assoc":  # Using 'Assoc' for Python 3.6 forward-ref compatibility
        """Return copy of Assoc without null values recorded.
        Usage:
            self.dropzeros()
            self.dropzeros(copy=True)
        Inputs:
            copy = (Optional, default False) Boolean indicating whether operation is 'in-place' or if a copy of the
                Assoc instance is made for which the null values are dropped.
        Outputs:
            Associative subarray of self consisting only of non-null values
        Notes:
            - Null values include 0, '', and None
        """
        # If numerical, just use scipy.sparse's eliminate_zeros() and condense
        if isinstance(self.val, float):
            if not copy:
                A = self
            else:
                A = self.deepcopy()

            # Remove zeros and update row and col appropriately
            _sparse.eliminate_zeros(A.adj)
            A.condense()
        # Otherwise, manually remove and remake Assoc instance
        else:
            assert isinstance(self.val, np.ndarray)
            if not copy:
                A = self
            else:
                A = self.deepcopy()

            null_val_indices = [
                index
                for index in range(len(self.val))
                if self.val[index] in Assoc.null_values
            ]

            # If there are no null values, immediately return original/copied associative array
            if len(null_val_indices) == 0:
                return A

            adj_row, adj_col, adj_data = _sparse.get_coo_params(self.adj)

            new_data = util.update_indices(
                adj_data, null_val_indices, len(self.val), offset=1, mark=0
            )
            if len(null_val_indices) == len(self.val):
                return Assoc([], [], [])
            else:
                A.val = np.delete(A.val, null_val_indices)

            adj_triples = zip(adj_row, adj_col, new_data)
            good_row_keys = set()
            good_col_keys = set()
            for triple in adj_triples:
                row_key, col_key, value = triple
                if value != 0:
                    good_row_keys.add(row_key)
                    good_col_keys.add(col_key)

            null_row_indices = [
                index for index in range(len(self.row)) if index not in good_row_keys
            ]
            if len(null_row_indices) == len(self.row):
                return Assoc([], [], [])
            else:
                new_row_indices = util.update_indices(
                    adj_row, null_row_indices, len(self.row), mark=0
                )
                A.row = np.delete(A.row, null_row_indices)

            null_col_indices = [
                index for index in range(len(self.col)) if index not in good_col_keys
            ]
            if len(null_col_indices) == len(self.col):
                return Assoc([], [], [])
            else:
                new_col_indices = util.update_indices(
                    adj_col, null_col_indices, len(self.col), mark=0
                )
                A.col = np.delete(A.col, null_col_indices)

            A.adj = _sparse.from_coo(
                new_row_indices, new_col_indices, new_data,
                dtype=int,
                shape=(len(A.row), len(A.col)),
            )
            _sparse.eliminate_zeros(A.adj)

        return A

    # Remove row/col indices that do not appear in the data
    def condense(self) -> "Assoc":
        """Remove items from self.row and self.col which do not correspond to values, according to self.adj.
        Usage:
            self.condense()
        Output:
            self = self.condense() = Associative array which removes all elements of self.row and self.col
                    which are not associated with some (nonzero) value.
        Notes:
            - In-place operation.
            - Elements of self.row or self.col which correspond to rows or columns of all 0's
                (but not '' or None) are removed.
        """
        row_dim, col_dim = _sparse.get_shape(self.adj)

        _, _, row_indptr = _sparse.get_csr_params(self.adj)
        good_rows = (row_indptr[:-1] < row_indptr[1:])
        self.row = self.row[:row_dim][good_rows]
        self.adj = _sparse.getitem_rows(self.adj, good_rows)

        _, _, col_indptr = _sparse.get_csc_params(self.adj)
        good_cols = (col_indptr[:-1] < col_indptr[1:])
        self.col = self.col[:col_dim][good_cols]
        self.adj = _sparse.getitem_cols(self.adj, good_cols)

        # Account for empty arrays
        if (len(self.row) == 0
                or len(self.col) == 0
                or _sparse.get_shape(self.adj)[0] == 0
                or _sparse.get_shape(self.adj)[1] == 0
        ):
            self.row = np.empty(0)
            self.col = np.empty(0)
            self.val = 1.0
            self.adj = _sparse.empty()

        return self

    # extension of condense() which also removes unused values
    def deepcondense(self) -> "Assoc":
        """Remove values from self.val which are not reflected in self.adj."""
        # If numerical, do nothing (no unused values)
        if isinstance(self.val, float):
            return self
        else:
            assert isinstance(self.val, np.ndarray)
            used_values = set([self.val[datum - 1] for datum in _sparse.get_data(self.adj)])
            print("used_values={}".format(used_values))
            unused_indices = [
                index
                for index in range(len(self.val))
                if self.val[index] not in used_values
            ]
            print("unused_indices={}".format(unused_indices))
            print("self_adj_data_pre_update={}".format(_sparse.get_data(self.adj)))
            updated_val_indices = util.update_indices(
                _sparse.get_data(self.adj), unused_indices, len(self.val), offset=1
            )
            print("updated_val_indices={}".format(updated_val_indices))
            print("self.adj before update=" + str(self.adj))
            self.adj = _sparse.update_data(self.adj, updated_val_indices)
            print("self.adj after update=" + str(self.adj))
            print("self.val before update=" + str(self.val))
            self.val = np.delete(self.val, unused_indices)
            print("self.val after update=" + str(self.val))

            # If self.val is now empty, make associative array empty
            if np.size(self.val) == 0:
                self.row = np.array([])
                self.col = np.array([])
                self.val = 1.0
                self.adj = _sparse.empty()

            return self

    def set_row(self, new_row: ArrayLike) -> "Assoc":
        """Replace current sorted array of row keys with new row keys. (in-place)
        Usage:
            self.set_row(new_row)
        Input:
            new_row = sequence of string or numerical values
        Output:
            self = Associative array with self.row replaced by np.unique(new_row) if new_row is compatible with
                the shape of self.adj
        Notes:
            - new_row is compatible with (row_dim, col_dim) = self.adj's shape if row_dim is at most the number of
                unique elements in new_row
        """
        new_row = util.sanitize(new_row)

        row_dim, _ = _sparse.get_shape(self.adj)
        true_row_size = len(set(new_row))
        if true_row_size < row_dim:
            raise ValueError("new_row is incompatible with the shape of self.adj.")
        if true_row_size == len(new_row) and util.np_sorted(new_row):
            self.row = new_row
        else:
            self.row = np.unique(new_row)

        # Remove unused row keys, column keys, and values
        self.condense()
        self.deepcondense()

        return self

    def set_col(self, new_col: ArrayLike) -> "Assoc":
        """Replace current sorted array of column keys with new column keys. (in-place)
        Usage:
            self.set_col(new_col)
        Input:
            new_col = sequence of string or numerical values
        Output:
            self = Associative array with self.col replaced by np.unique(new_col) if new_col is compatible with
                the shape of self.adj
        Notes:
            - new_col is compatible with (row_dim, col_dim) = self.adj's shape if col_dim is at most the number of
                unique elements in new_col
        """
        new_col = util.sanitize(new_col)

        _, col_dim = _sparse.get_shape(self.adj)
        true_col_size = len(set(new_col))
        if true_col_size < col_dim:
            raise ValueError("new_col is incompatible with the shape of self.adj.")
        if true_col_size == len(new_col) and util.np_sorted(new_col):
            self.col = new_col
        else:
            self.col = np.unique(new_col)

        # Remove unused row keys, column keys, and values
        self.condense()
        self.deepcondense()

        return self

    def set_val(self, new_val: ArrayLike) -> "Assoc":
        """Replace current sorted array of unique values with new values. (in-place)
        Usage:
            self.set_val(new_val)
        Input:
            new_val = value or sequence of values
        Output:
            self = Associative array with np.unique(new_val) replacing self.val if non-numerical or self.adj's data
                if numerical, where self.adj's data is treated as indices used to select from np.unique(new_val)
        Notes:
            - self.adj's data is converted to dtype=int to treat as indices
        """
        if new_val in Assoc.null_values:
            self.row = np.array([])
            self.col = np.array([])
            self.val = 1.0
            self.adj = _sparse.empty()
            return self
        else:
            new_val = util.sanitize(new_val)

            # Remove any null values from new_val
            for null_value in Assoc.null_values:
                if np.issubdtype(new_val.dtype, type(null_value)):
                    new_val = new_val[np.where(new_val != null_value)]

        _sparse.eliminate_zeros(self.adj)
        adj_row, adj_col, adj_data = _sparse.get_coo_params(self.adj)
        if len(new_val) == 1:
            triple_num = _sparse.nnz(self.adj)
            if np.issubdtype(new_val.dtype, int) or np.issubdtype(new_val.dtype, float):
                self.val = 1.0
                self.adj = _sparse.from_coo(adj_row, adj_col, np.full(triple_num, new_val[0]), dtype=float)
            else:
                self.val = new_val
                self.adj = _sparse.from_coo(adj_row, adj_col, np.ones(triple_num), dtype=int)
        else:
            # Treat self.adj as containing indices of entries in new_val
            true_val_size = len(set(new_val))
            data_indices = adj_data.astype(int) - 1
            min_index, max_index = data_indices.min(), data_indices.max()
            if not (0 <= min_index and max_index < true_val_size):
                raise ValueError("new_val is incompatible with the data in self.adj.")

            if true_val_size != len(new_val) or not util.np_sorted(new_val):
                new_val = np.unique(new_val)

            if np.issubdtype(new_val.dtype, int) or np.issubdtype(new_val.dtype, float):
                self.val = 1.0
                self.adj = _sparse.from_coo(adj_row, adj_col, new_val[data_indices], dtype=float)
            else:
                self.val = new_val[0: (max_index + 1)]
                self.adj = _sparse.from_coo(adj_row, adj_col, data_indices + 1, dtype=int)

        # Remove unused row keys, column keys, and values
        self.condense()
        self.deepcondense()

        return self

    def set_adj(self, new_adj: _sparse.Sparse_Matrix, numerical: bool = True) -> "Assoc":
        """Replace current adjacency array with new adjacency array. (in-place)
        Usage:
            self.set_adj(new_adj)
        Input:
            new_adj = A sparse matrix whose dimensions are at most that of self.adj
            numerical = (Optional, default True) Boolean indicating if the resulting associative array
                should be treated as numerical, with new_adj containing the new (numerical) values. If
                numerical is set to False, then the values of new_adj are treated as indices of the actual
                values, stored in self.val, in which case floats are converted to ints as needed.
        Output:
            self = Associative array with given sparse matrix as adjacency array
                and row and column values cut down to fit the dimensions of the
                new adjacency array
        """
        if numerical is None:
            numerical = True

        row_size, col_size = _sparse.get_shape(new_adj)
        if len(self.row) < row_size or len(self.col) < col_size:
            raise ValueError(
                "The shape of new_adj is incompatible with the sizes of self.row and/or self.col."
            )

        if numerical:
            self.val = 1.0
            self.adj = _sparse.to_coo(new_adj)
        else:
            self.adj = _sparse.astype(_sparse.to_coo(new_adj), int)

        # Remove unused row keys, column keys, and values
        self.condense()
        self.deepcondense()

        return self

    putrow, putcol, putval, putadj = set_row, set_col, set_val, set_adj
    put_row, put_col, put_val, put_adj = set_row, set_col, set_val, set_adj

    def __setitem__(self, key: Tuple[KeyVal, KeyVal], value: KeyVal) -> None:
        """Add the triple (row_key, col_key, value) to associative array."""
        row_key, col_key = key
        if len(self.row) == 0:
            # Handle empty associative array case
            self.row = np.array([row_key])
            self.col = np.array([col_key])

            if util.is_numeric(value):
                self.val = 1.0
                self.adj = _sparse.from_coo(np.array([0]), np.array([0]), np.array([value]))
            else:
                self.val = np.array([value])
                self.adj = _sparse.from_coo(np.array([0]), np.array([0]), np.array([1]), dtype=int)

            return None
        else:
            adj_row, adj_col, adj_data = _sparse.get_coo_params(self.adj)
            self.row, new_adj_row = util.sorted_append(
                row_key,
                self.row,
                adj_row,
                new_entry_name="row_key",
                sorted_array_name="self.row",
            )
            self.col, new_adj_col = util.sorted_append(
                col_key,
                self.col,
                adj_col,
                new_entry_name="col_key",
                sorted_array_name="self.col",
            )

            # Reconstruct self.adj so shape and dtype are correct
            if isinstance(self.val, float) and (
                isinstance(value, int) or isinstance(value, float)
            ):
                new_adj_data = np.append(adj_data, value).astype(float)
                self.adj = _sparse.from_coo(new_adj_row, new_adj_col, new_adj_data)
            else:
                adjusted_data = (
                    _sparse.get_data(self.adj) - 1
                )  # Decrement so indices are properly zero-indexed
                self.val, new_adj_data = util.sorted_append(
                    value,
                    self.get_val(),
                    adjusted_data,
                    new_entry_name="value",
                    sorted_array_name="self values",
                )
                new_adj_data += 1  # Increment indices to one-index
                self.adj = _sparse.from_coo(new_adj_row, new_adj_col, new_adj_data, dtype=int)

            return None

    def find(
        self, ordering: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get row, col, and val arrays that would generate the Assoc (reverse constructor).
        Usage:
            self.find()
        Input:
            ordering = (Optional, default None)
                - if 0, then order by row first, then column
                - if 1, then order by column first, then row
                - if None, no particular order is guaranteed
        Output:
            row,col,val = numpy arrays for which self = Assoc(row, col, val)
        """
        if is_empty_assoc(self):
            return np.array([]), np.array([]), np.array([])
        else:
            pass

        enc_row, enc_col, enc_val = list(), list(), list()
        # Use self.adj to extract row, col, and val
        if ordering == 0:
            # use csr format parameters to order by row first, then column
            data, indices, indptr = _sparse.get_csr_params(self.adj)
            lower_row_index = 0
            for row_index in range(1, np.size(indptr)):
                upper_row_index = indptr[row_index]
                for val_index in range(lower_row_index, upper_row_index):
                    enc_row.append(row_index-1)
                    enc_col.append(indices[val_index])
                    enc_val.append(data[val_index])
                lower_row_index = upper_row_index
        elif ordering == 1:
            # use csc format parameters to order by column first, then row
            data, indices, indptr = _sparse.get_csc_params(self.adj)
            lower_col_index = 0
            for col_index in range(1, np.size(indptr)):
                upper_col_index = indptr[col_index]
                for val_index in range(lower_col_index, upper_col_index):
                    enc_row.append(indices[val_index])
                    enc_col.append(col_index-1)
                    enc_val.append(data[val_index])
                lower_col_index = upper_col_index
        else:
            # use coo format parameters to get /some/ order
            enc_row, enc_col, enc_val = _sparse.get_coo_params(self.adj)

        # enc_row, enc_col are arrays of index pointers to self.row, self.col
        row, col = self.row[enc_row], self.col[enc_col]

        # enc_val is an array of index pointers to self.val in nonnumerical case and the values in the numerical case
        if isinstance(self.val, float):
            val = enc_val
        else:
            assert isinstance(self.val, np.ndarray)
            enc_val = [int(item - 1) for item in enc_val]
            if np.size(enc_val) != 0:
                val = self.val[enc_val]
            else:
                val = np.array([])

        return row, col, val

    def triples(
        self, ordering: Optional[int] = None
    ) -> List[Tuple[KeyVal, KeyVal, KeyVal]]:
        """Return list of triples of form (row_label,col_label,value)."""
        row, col, val = self.find(ordering=ordering)
        return list(zip(row, col, val))

    def to_dict(self) -> Dict[KeyVal, Dict[KeyVal, KeyVal]]:
        """Return a dictionary satisfying self.to_dict[row_key][col_key] = value if and only if row_key col_key,
        value correspond to an entry of self."""
        row, col, val = self.find()
        adjacency_dict = dict()

        for index in range(np.size(row)):
            if row[index] not in adjacency_dict:
                adjacency_dict[row[index]] = {col[index]: val[index]}
            else:
                adjacency_dict[row[index]][col[index]] = val[index]

        return adjacency_dict

    def to_dict2(self) -> Dict[Tuple[KeyVal, KeyVal], KeyVal]:
        """Return a dictionary satisfying self.to_dict2[(row_key, col_key)] = value if and only if row_key, col_key,
        value correspond to an entry of self."""
        row, col, val = self.find()
        return dict(zip(zip(row, col), val))

    def get_row(self) -> np.ndarray:
        return self.row

    def get_col(self) -> np.ndarray:
        return self.col

    def get_val(self) -> np.ndarray:
        """Return numpy array of unique values."""
        if isinstance(self.val, float):
            return np.unique(_sparse.get_data(self.adj))
        else:
            assert isinstance(self.val, np.ndarray)
            return self.val

    def get_adj(self) -> _sparse.Sparse_Matrix:
        return self.adj

    def get_value(
        self,
        row_key: KeyVal,
        col_key: KeyVal,
        indices: bool = False,
        row_index: bool = False,
        col_index: bool = False,
    ) -> KeyVal:
        """Get the value in self corresponding to given row_key and col_key, otherwise return 0.
        Usage:
            v = self.getvalue('a', 'B')
        Inputs:
            row_key = row key (or index, see optional argument 'indices' below)
            col_key = column key (or index, see optional argument 'indices' below)
            indices = (Optional, default False) Boolean indicating whether row_key and col_key should be
                interpreted as indices instead of *actual* row and column keys, respectively.
            row_index = (Optional, default False) 'indices', but only affecting the interpretation of row_key
            col_index = (Optional, default False) 'indices', but only affecting the interpretation of col_key
        Output:
            v = value of self corresponding to the pair (row_key, col_key),
                i.e., (row_key, col_key, v) is in self.triples()
        Note:
            'indices' supersedes 'row_index' and 'col_index', so if indices=True, then row_index and col_index are
                set to True, regardless of if/how those latter optional arguments were set.
        """
        if indices:
            row_index = True
            col_index = True

        row_size, col_size = _sparse.get_shape(self.adj)

        if not row_index:
            try:
                row_key = np.where(self.row == row_key)[0][0]
            except IndexError:
                return 0
        else:
            assert isinstance(row_key, int)
            if row_key < 0 or row_size < row_key:
                return 0
        if not col_index:
            try:
                col_key = np.where(self.col == col_key)[0][0]
            except IndexError:
                return 0
        else:
            assert isinstance(col_key, int)
            if col_key < 0 or col_size < col_key:
                return 0

        if isinstance(self.val, float):
            return _sparse.getvalue(self.adj, row_key, col_key)
        else:
            assert isinstance(self.val, np.ndarray)
            value_index = _sparse.getvalue(self.adj, row_key, col_key) - 1
            return self.val[value_index]

    # Overload getitem; allows for subsref
    def __getitem__(self, selection: Tuple[Selectable, Selectable]) -> "Assoc":
        """Returns a sub-associative array of self according to object1 and object2 or corresponding value
        Usage:
            B = self[row_select, col_select]
        Inputs:
            selection = tuple (row_select, col_select) where
                row_select = string of (delimiter separated) values (delimiter is last character)
                    or iterable or int or slice object or function
                col_select = string of (delimiter separated) values (delimiter is last character)
                    or iterable or int or slice object or function
                    e.g., "a,:,b,", "a,b,c,d,", ['a',':','b'], 3, [1,2], 1:2, startswith("a,b,"), :, ':,c,'
        Outputs:
            B = sub-associative array of self whose row indices are selected by row_select and whose
                column indices are selected by col_select, assuming not both of row_select, col_select are single
                indices
            B = value of self corresponding to single indices of row_select and col_select
        Examples:
            self['a,:,b,', ['0', '1']]
            self[1:2:1, 1]
        Note:
            - Regular slices are NOT right end-point inclusive
            - 'Slices' of the form "a,:,b," ARE right end-point inclusive (i.e., includes b)
            - Integer row_select or col_select, and by extension slices, do not reference actual entries in
                self.row or self.col, but the induced indexing of the rows and columns
                e.g., self[:, 0:2] will give the subarray consisting of all rows and the columns col[0], col[1],
                self[:, 0] will give the subarray consisting of the 0-th column
                self[2, 4] will give the value in the 2-nd row and 4-th column
        """
        row_select, col_select = selection
        new_row, row_index_map = util.select_items(
            row_select, self.row, return_indices=True
        )
        new_col, col_index_map = util.select_items(
            col_select, self.col, return_indices=True
        )

        self_adj_sub = _sparse.getitem_rows(self.adj, row_index_map)
        print("self_adj_sub row first:{}".format(self_adj_sub))
        self_adj_sub = _sparse.getitem_cols(self_adj_sub, col_index_map)
        print("self_adj_sub column second:{}".format(self_adj_sub))
        self_adj_sub = _sparse.coo_canonicalize(self_adj_sub)
        subarray = Assoc(
            np.array(new_row),
            np.array(new_col),
            self.val,
            self_adj_sub,
            aggregate="unique",
        )
        print(subarray)
        subarray = subarray.condense()
        print(subarray)
        subarray = subarray.deepcondense()
        print(subarray)
        return subarray

    def size(self) -> Tuple[int, int]:
        """Returns dimensions of self."""
        size1 = np.size(self.row)
        size2 = np.size(self.col)
        return size1, size2

    def nnz(self) -> int:
        """Count number of non-null entries."""
        num_nonzero = _sparse.nnz(self.adj)
        return num_nonzero

    def __str__(self) -> str:
        """Print formatted attributes."""
        print_string = "Row indices: " + str(self.row) + "\n"
        print_string += "Column indices: " + str(self.col) + "\n"
        print_string += "Values: " + str(self.val) + "\n"
        print_string += "Adjacency array: " + "\n" + str(_sparse.to_array(self.adj))
        return print_string

    # print tabular form
    def printfull(self, cutoff_rows: bool = False, cutoff_cols: bool = False) -> None:
        """Print in tabular form.
        Usage:
            A.printfull()
            A.printfull(cutoff_rows=True)
            A.printfull(cutoff_cols=True)
        Inputs:
            cutoff_rows = (Optional, default False) Boolean indicating whether the number of rows should be restricted
                to fit within the terminal window height
            cutoff_cols = (Optional, default False) Boolean indicating whether the number of columns should be
                restricted so that the total width does not exceed the terminal window's width
        Outputs: N/A
        """
        if (isinstance(self.val, float) and np.size(_sparse.get_data(self.adj)) == 0) or np.size(
                self.val
        ) == 0:
            print("Empty associative array.")
            return None

        max_array_rows, max_array_cols = self.size()  # values may change if cutoff rows or cols
        terminal_col, terminal_row = shutil.get_terminal_size()  # in case cutoff_rows or cutoff_cols

        # update rows of interest based on terminal size if cutoff_rows
        need_cutoff_row = False
        if cutoff_rows:
            # Determine if all rows fit in terminal window (with col labels and array size footer)
            if np.size(self.row) <= terminal_row - 3:
                max_array_rows = np.size(self.row)
            else:
                need_cutoff_row = True
                max_array_rows = (
                        terminal_row - 5
                )  # Include space for vertical ellipses and final row

            rel_array = self[0:max_array_rows, :]  # Disregard rows outside range
            trans_dict = (
                rel_array.transpose().to_dict()
            )  # Take transpose to collate by column labels

            # Add in final row (may already be present)
            final_row_dict = self[-1, :].transpose().to_dict()
            for column_key in trans_dict.keys():
                if column_key in final_row_dict.keys():
                    trans_dict[column_key].update(final_row_dict[column_key])
        else:
            rel_array = self
            trans_dict = (
                self.transpose().to_dict()
            )  # Take transpose to collate by column labels

        # Find widths of columns (w.r.t. only pre-cutoff rows and last row, unless column is empty as a result)
        col_widths = list()
        for index in np.arange(np.size(self.col)):
            col_label = self.col[index]
            if col_label in trans_dict:
                width = max(
                    [len(str(val)) for val in trans_dict[col_label].values()]
                )
            else:
                width = 0
            width = max([width, len(str(col_label))])
            col_widths.append(width)

        # Format array values according to calculated column-wise max widths
        if cutoff_rows:
            rel_dict = rel_array.to_dict()
            rel_dict.update(
                self[-1, :].to_dict()
            )  # Add in final row (may already be present)
        else:
            rel_dict = self.to_dict()
        formatted_rows = list()
        row_indices = list(range(max_array_rows))
        if cutoff_rows and need_cutoff_row:
            row_indices.append(-1)
        for row_index in row_indices:
            current_row = list()
            for col_index in np.arange(np.size(self.col)):
                if self.col[col_index] in rel_dict[self.row[row_index]]:
                    row_item = str(
                        rel_dict[self.row[row_index]][self.col[col_index]]
                    )
                else:
                    row_item = ""
                current_row.append(
                    ("  {:>" + str(col_widths[col_index]) + "}").format(row_item)
                )
            formatted_rows.append(current_row)
        if cutoff_rows and need_cutoff_row:
            vellipses = list()
            for col_index in np.arange(np.size(self.col)):
                vellipses.append(
                    ("  {:>" + str(col_widths[col_index]) + "}").format(":")
                )  # ⋮
            formatted_rows.insert(-1, vellipses)

        # Format column labels according to calculated column-wise max widths
        col_labels = list()
        for index in np.arange(np.size(self.col)):
            col_labels.append(
                ("  {:>" + str(col_widths[index]) + "}").format(
                    str(self.col[index])
                )
            )

        # Format row labels according to max (relevant) row label width
        row_label_width = max(
            [len(str(self.row[row_index])) for row_index in row_indices]
        )
        row_labels = list()
        for row_index in row_indices:
            row_labels.append(
                ("{:<" + str(row_label_width) + "}").format(
                    str(self.row[row_index])
                )
            )
        if cutoff_rows and need_cutoff_row:
            row_labels.insert(-1, ("{:<" + str(row_label_width) + "}").format(":"))

        # Determine how many columns fit in terminal window and print
        last_index = np.size(self.col) - 1
        too_wide = False

        if not cutoff_cols:
            print(" " * row_label_width + "".join(col_labels))
            for row_index in np.arange(len(row_labels)):
                print(row_labels[row_index] + "".join(formatted_rows[row_index]))
        else:
            # Case 1: All columns fits
            if row_label_width + sum(col_widths) + 2 * (last_index + 1) <= terminal_col:
                print(" " * row_label_width + "".join(col_labels))
                for row_index in np.arange(len(row_labels)):
                    print(row_labels[row_index] + "".join(formatted_rows[row_index]))

            # Case 2: Not all columns fit, but at least the first and last do, plus ellipsis
            elif (
                    row_label_width + col_widths[0] + col_widths[last_index] + 9
                    <= terminal_col
            ):

                # Determine how many columns fit
                penul_index = 0
                running_total = row_label_width + col_widths[last_index] + 9
                while running_total + col_widths[penul_index] + 2 <= terminal_col:
                    running_total += col_widths[penul_index] + 2
                    penul_index += 1

                print(
                    " " * row_label_width
                    + "".join(col_labels[0:penul_index])
                    + "  ..."
                    + col_labels[last_index]
                )
                for row_index in np.arange(len(row_labels)):
                    print(
                        row_labels[row_index]
                        + "".join(formatted_rows[row_index][0:penul_index])
                        + "  ..."
                        + formatted_rows[row_index][last_index]
                    )

                too_wide = True

            # Case 3: First column plus ellipses fit, but not with addition of final column
            elif row_label_width + col_widths[0] + 7 <= terminal_col:

                # Determine how many columns fit
                penul_index = 0
                running_total = row_label_width + 7
                while running_total + col_widths[penul_index] + 2 <= terminal_col:
                    running_total += col_widths[penul_index] + 2
                    penul_index += 1

                print(
                    " " * row_label_width + "".join(col_labels[0:penul_index]) + "  ..."
                )
                for row_index in np.arange(len(row_labels)):
                    print(
                        row_labels[row_index]
                        + "".join(formatted_rows[row_index][0:penul_index])
                        + "  ..."
                    )

                too_wide = True

            # Case 4: Can't fit even the first column plus ellipsis
            else:
                print("Columns too wide to fit window.")

                too_wide = True

        # Report dimensions if either horizontally or vertically cut off
        if (cutoff_rows and need_cutoff_row) or too_wide:
            print(
                "\n"
                + "["
                + str(np.size(self.row))
                + " rows x "
                + str(np.size(self.col))
                + " columns]"
            )

        return None

    def spy(self, rename_axes: bool = False, **pyplot_spy_kwargs) -> plt.Line2D:
        """Print spy plot of self.adj"""

        def bounded_interpolator(x, min_point, max_point):
            x0, y0 = min_point
            x1, y1 = max_point
            return min(max(((y1 - y0) / (x1 - x0)) * (x - x0) + y0, y0), y1)

        marker_size = bounded_interpolator(
            max(len(self.row), len(self.col)), (1000, 0.1), (1, 3)
        )

        assoc_spy = plt.spy(
            _sparse.to_scipy(self.adj), markersize=marker_size, aspect="auto", **pyplot_spy_kwargs
        )

        if rename_axes:
            row_font_size = bounded_interpolator(len(self.row), (1000, 0.1), (1, 10))
            col_font_size = bounded_interpolator(len(self.col), (1000, 0.1), (1, 10))

            plt.xticks(
                range(len(self.row)), self.row, fontsize=row_font_size, rotation=45
            )
            plt.yticks(range(len(self.col)), self.col, fontsize=col_font_size)

        plt.show()
        return assoc_spy

    def copy(self) -> "Assoc":
        """Create a shallow copy of self."""
        assoc_copy = Assoc(
            cpy.copy(self.row),
            cpy.copy(self.col),
            cpy.copy(self.val),
            _sparse.copy(self.adj),
            aggregate="unique",
        )
        return assoc_copy

    def __copy__(self) -> "Assoc":
        return self.copy()

    def deepcopy(self) -> "Assoc":
        """Create a deep copy of self."""
        assoc_deepcopy = Assoc(
            cpy.deepcopy(self.row),
            cpy.deepcopy(self.col),
            cpy.deepcopy(self.val),
            cpy.deepcopy(self.adj),
            aggregate="unique",
        )
        return assoc_deepcopy

    def __deepcopy__(self) -> "Assoc":
        return self.deepcopy()

    def diag(self):
        """Extract the diagonal of self as an associative array."""
        # Get intersection of self.row and self.col
        diagonal_keys, row_map, col_map = util.sorted_intersect(
            self.row, self.col, return_index=True
        )

        # Build sub-matrix of self.adj by element-wise multiplying by 0,1-valued sparse matrix picking out true diagonal
        dim = len(row_map)
        if dim == 0:
            return Assoc([], [], [])
        else:
            offset_diagonal = _sparse.from_coo(row_map, col_map, np.ones(dim), dtype=int)
            offset_square = _sparse.ewise_mult(self.adj, offset_diagonal)
            _sparse.eliminate_zeros(offset_square)  # Just to be safe

            # Extract the row and col indices that actually appear among diagonal entries
            offset_square_row, offset_square_col, offset_square_data = _sparse.get_coo_params(offset_square)
            square_row, index_map = np.unique(offset_square_row, return_index=True)
            square_col = offset_square_col[
                index_map
            ]  # Same index_map works for offset_square.col & offset_square.data
            if isinstance(self.val, float):
                square_data = offset_square_data[index_map]
                diag_val = 1.0
                diag_dtype = float
            else:
                assert isinstance(self.val, np.ndarray)
                square_data = index_map + 1
                diag_val = np.unique(self.val[offset_square_data[index_map] - 1])
                diag_dtype = int

            dim = len(square_row)
            diag_row, diag_col = (
                self.row[square_row],
                self.col[square_col],
            )  # Extract used row & column keys
            diag_adj = _sparse.from_coo(np.arange(dim), np.arange(dim), square_data, dtype=diag_dtype, shape=(dim, dim))

            return Assoc(diag_row, diag_col, diag_val, diag_adj, aggregate="unique")

    # replace all non-zero values with ones
    def logical(self, copy: bool = True) -> "Assoc":
        """Replaces every non-zero value with 1.0
        Usage:
            self.logical()
            self.logical(copy=False)
        Input:
            copy = (Optional, default True) Boolean indicating whether the operation is in-place or not
        Output:
            self.logical() = a copy of self with all non-zero values replaced with 1.0
            self.logical(copy=False) = self with all non-zero values replaced with 1.0
        """
        A = self.dropzeros(copy=copy)
        A.val = 1.0
        A.adj = _sparse.logical(A.adj)
        return A

    def transpose(self, copy: bool = True) -> "Assoc":
        """Transpose array, switching self.row and self.col and transposing self.adj."""
        if copy:
            transposed = Assoc(
                cpy.copy(self.col),
                cpy.copy(self.row),
                cpy.copy(self.val),
                _sparse.transpose(self.adj, copy=True),
                aggregate="unique",
            )
        else:
            self.row, self.col = self.col, self.row
            self.adj = _sparse.transpose(self.adj)
            transposed = self
        return transposed

    # Eliminate columns
    def nocol(self, copy: bool = True) -> "Assoc":
        """Eliminate columns.
        Usage:
            self.nocol()
            self.nocol(copy=False)
        Input:
            copy = (Optional, default True) Boolean indicating whether operation should be in-place
        Output:
            self.nocol() = Associative array with same row indices as self and single column index 0.
                        The i-th row of self.nocol() is 1 only when the i-th row of self had a non-zero entry.
            self.nocol(copy=False) = in-place version
        """
        if copy:
            self_ = self.deepcopy()
        else:
            self_ = self

        # Condense and extract remaining row keys
        self_.dropzeros()
        self_.condense()
        if len(self_.row) == 0 or len(self_.col) == 0:
            return Assoc([], [], [])

        length = len(self_.row)
        self_.col = np.array([0])
        self_.val = 1.0
        self_.adj = _sparse.from_coo(np.arange(0, length, dtype=int), np.zeros(length, dtype=int), np.ones(length))

        return self_

    # Eliminate rows
    def norow(self, copy: bool = True):
        """Eliminate rows.
        Usage:
            self.norow()
            self.norow(copy=False)
        Input:
            copy = (Optional, default True) Boolean indicating whether operation should be in-place
        Output:
            self.norow() = Associative array with same col indices as self and single row index 0.
                        The i-th col of self.norow() is 1 only when the i-th col of self had a non-zero entry.
            self.norow(copy=False) = in-place version
        """
        if copy:
            self_ = self.deepcopy()
        else:
            self_ = self

        # Condense and extract remaining row keys
        self_.dropzeros()
        self_.condense()
        if len(self_.row) == 0 or len(self_.col) == 0:
            return Assoc([], [], [])

        length = len(self_.col)
        self_.row = np.array([0])
        self_.val = 1.0
        self_.adj = _sparse.from_coo(np.zeros(length, dtype=int), np.arange(0, length, dtype=int), np.ones(length))

        return self_

    def sum(self, axis: Optional[int] = None) -> Union[float, "Assoc"]:
        """Sum over the given axis or over whole array if None.
        Usage:
            self.sum()
            self.sum(0)
            self.sum(1)
        Input:
            axis = 0 if summing down columns, 1 if summing across rows, and None if over whole array
        Output:
            self.sum() = sum of all entries in self (or self.nnz() if self has non-numerical entries)
            self.sum(axis) = Associative array resulting from summing over indicated axis
        """
        # If any of the values are strings, convert to logical
        # In this case, the adjacency array is the desired sparse matrix to sum over
        if not isinstance(self.val, float):
            new_sparse = _sparse.copy(self.logical(copy=True).adj)
        # Otherwise, build a new sparse matrix with actual (numerical) values
        else:
            new_sparse = _sparse.copy(self.adj)

        # Sum as a sparse matrix over desired axis
        summed_sparse = _sparse.sparse_sum(new_sparse, axis)

        # Depending on axis, build associative array
        if axis is None:
            return summed_sparse
        elif axis == 1:
            summed_sparse_adj = _sparse.transpose(_sparse.init([summed_sparse]))
            A = Assoc(
                self.row.copy(),
                np.array([0]),
                1.0,
                summed_sparse_adj,
                aggregate="unique",
            )
        elif axis == 0:
            summed_sparse_adj = _sparse.init([summed_sparse])
            A = Assoc(
                np.array([0]),
                self.col.copy(),
                1.0,
                summed_sparse_adj,
                aggregate="unique",
            )
        else:
            A = None

        A.condense()
        A.deepcondense()

        return A

    def combine(
        self,
        other: "Assoc",
        binary_op: Callable[[KeyVal, KeyVal], KeyVal],
        right_zero: bool = False,
        left_zero: bool = False,
        zero: Optional[KeyVal] = None,
    ) -> "Assoc":
        """Generic method for combining two associative arrays according to a given binary operation binary_op.
        Inputs:
            other = associative array
            binary_op = binary operation compatible with the entries in self & other
            right_zero = (Optional, default False) Boolean indicating whether the operation
                'binary_op(value, 0)' must occur for values corresponding to entries in self
            left_zero = (Optional, default False) Boolean indicating whether the operation
                'binary_op(0, value)' must occur for values corresponding to entries in other
            zero = (Optional, default None) modifies the computations in right_zero and/or left_zero to
                'binary_op(value, zero)' and/or 'binary_op(zero, value)', respectively. E.g., zero = ''. If
                zero is None, then the value of None is inferred from the dtypes of self and other.
        Output:
            self.combine(other, binary_op) = associative array combining entries from both associative arrays
                according to the binary operation binary_op, with optional added support for handling
                entries of self or other which do not have corresponding entries in other or self, resp.
        Notes:
            - If the optional Booleans right_zero and/or left_zero are utilized, binary_op must support the
                computations 'binary_op(value, zero)' and/or 'binary_op(zero, value)', respectively, where zero is
                inferred from the dtypes of self.val and other.val (zero='' if strings, 0 if numerical) if not
                explicitly given.
            - If one of self.val and other.val has string data while the other has numerical data, the numerical
                data is automatically converted into string data.
        """
        self_row, self_col, self_val = self.find()
        other_row, other_col, other_val = other.find()
        if (
            (
                np.issubdtype(self_row.dtype, other_row.dtype)
                or np.issubdtype(other_row.dtype, self_row.dtype)
            )
            and (
                np.issubdtype(self_col.dtype, other_col.dtype)
                or np.issubdtype(other_col.dtype, self_col.dtype)
            )
            and (
                np.issubdtype(self_val.dtype, other_val.dtype)
                or np.issubdtype(other_val.dtype, self_val.dtype)
            )
        ):
            pass
        else:
            warnings.warn(
                "Combining associative arrays whose rows, cols, or vals have incompatible dtypes may result"
                "in silent upcasting, e.g., float + str -> str."
            )
        new_row = np.append(self_row, other_row)
        new_col = np.append(self_col, other_col)
        new_val = np.append(self_val, other_val)

        if zero is None:
            if np.issubdtype(new_val.dtype, np.number):
                zero = 0
            else:
                zero = ""

        if right_zero or left_zero:
            self_dict = self.to_dict2()
            other_dict = other.to_dict2()
            if right_zero:
                # Post-append triples from self (with no corresponding triple in other) with values set to 0
                self_zero_keys = [
                    key_pair
                    for key_pair in self_dict.keys()
                    if key_pair not in other_dict.keys()
                ]
                self_zero_row = np.array([key_pair[0] for key_pair in self_zero_keys])
                self_zero_col = np.array([key_pair[1] for key_pair in self_zero_keys])
                new_row = np.append(new_row, self_zero_row)
                new_col = np.append(new_col, self_zero_col)
                new_val = np.append(new_val, np.full(len(self_zero_keys), zero))
            if left_zero:
                # Pre-append triples from other (with no corresponding triple in self) with values set to 0
                other_zero_keys = [
                    key_pair
                    for key_pair in other_dict.keys()
                    if key_pair not in self_dict.keys()
                ]
                other_zero_row = np.array([key_pair[0] for key_pair in other_zero_keys])
                other_zero_col = np.array([key_pair[1] for key_pair in other_zero_keys])
                new_row = np.append(other_zero_row, new_row)
                new_col = np.append(other_zero_col, new_col)
                new_val = np.append(np.full(len(other_zero_keys), zero), new_val)

        return Assoc(new_row, new_col, new_val, aggregate=binary_op)

    def semiring_prod(
        self,
        other: "Assoc",
        semi_add: Callable[[KeyVal, KeyVal], KeyVal],
        semi_mult: Callable[[KeyVal, KeyVal], KeyVal],
    ) -> "Assoc":
        """Array multiplication taken with respect to given semiring addition and semiring multiplication operations.
        Note:
            - semi_add and semi_mult are assumed to obey the semiring axioms, i.e., semi_add is assumed to be
                commutative, associative, has abstract null as identity; semi_mult is assumed to be associative,
                has abstract null as annihilator, and distributes over semi_add.
        """
        # Not fully implemented
        # TODO: Decide how to handle semi_add's identity not being considered null w.r.t. semi_mult computations.

        # Intersect self.col and other.row
        intersection, index_map_1, index_map_2 = util.sorted_intersect(
            self.col, other.row, return_index=True
        )
        self_trimmed = self[:, index_map_1]
        other_trimmed = other[index_map_2, :]

        self_trimmed_adj = self_trimmed.adj
        other_trimmed_adj = other_trimmed.adj

        _, self_csr_indices, self_csr_indptr = _sparse.get_csr_params(self_trimmed_adj)
        _, other_csc_indices, other_csc_indptr = _sparse.get_csc_params(other_trimmed_adj)

        row_size = len(self_trimmed.row)
        col_size = len(other_trimmed.col)

        rows = [
            self_csr_indices[
                self_csr_indptr[row_index]: self_csr_indptr[
                    (row_index + 1)
                ]
            ]
            for row_index in range(row_size)
        ]
        cols = [
            other_csc_indices[
                other_csc_indptr[col_index]: other_csc_indptr[
                    (col_index + 1)
                ]
            ]
            for col_index in range(col_size)
        ]

        row_sets = [set(row) for row in rows]
        col_sets = [set(col) for col in cols]

        new_row = list()
        new_col = list()
        new_val = list()

        for row_index in range(row_size):
            for col_index in range(col_size):
                intersect = row_sets[row_index].intersection(col_sets[col_index])
                if len(intersect) != 0:
                    indices = np.intersect1d(
                        rows[row_index], cols[col_index], assume_unique=True
                    )
                    sum_prod = 0
                    for i in range(len(indices)):
                        inter_index = indices[i]

                        if isinstance(self_trimmed.val, float):
                            row_value = _sparse.getvalue(self_trimmed_adj, row_index, inter_index)
                        else:
                            assert isinstance(self_trimmed.val, np.ndarray)
                            row_value = self_trimmed.val[
                                _sparse.getvalue(self_trimmed_adj, row_index, inter_index) - 1
                            ]

                        if isinstance(other_trimmed.val, float):
                            col_value = _sparse.getvalue(other_trimmed_adj, inter_index, col_index)
                        else:
                            assert isinstance(other_trimmed.val, np.ndarray)
                            col_value = other_trimmed.val[
                                _sparse.getvalue(other_trimmed_adj, inter_index, col_index) - 1
                            ]

                        sum_prod = semi_add(sum_prod, semi_mult(row_value, col_value))

                    new_row.append(self_trimmed.row[row_index])
                    new_col.append(other_trimmed.col[col_index])
                    new_val.append(sum_prod)
                else:
                    pass

        return Assoc(new_row, new_col, new_val)

    # Overload element-wise addition
    def __add__(self, other: "Assoc") -> "Assoc":
        """Element-wise addition of self and other, matched up by row and column indices.
        Usage:
            self + other
        Input:
            other = Associative array
        Output:
            self + other =
                - element-wise sum of self and other (if both are numerical)
                - element-wise concatenation of entries of self and other (if neither are numerical)
        Note:
            - If one argument is non-numerical and the other is, the non-numerical array has .logical() called
                prior to addition. This may produce undesired results!
        """
        if isinstance(self.val, float) and isinstance(other.val, float):
            # Take union of rows and cols while keeping track of indices
            row_union, row_index_self, row_index_other = util.sorted_union(self.row, other.row, return_index=True)
            col_union, col_index_self, col_index_other = util.sorted_union(self.col, other.col, return_index=True)
            self_adj_row, self_adj_col, self_adj_data = _sparse.get_coo_params(self.adj)
            other_adj_row, other_adj_col, other_adj_data = _sparse.get_coo_params(other.adj)
            Aadj_reindex = _sparse.from_coo(row_index_self[self_adj_row],
                                            col_index_self[self_adj_col],
                                            self_adj_data,
                                            shape=(len(row_union), len(col_union)),
                                            dtype=float
                                            )
            Badj_reindex = _sparse.from_coo(row_index_other[other_adj_row],
                                            col_index_other[other_adj_col],
                                            other_adj_data,
                                            shape=(len(row_union), len(col_union)),
                                            dtype=float
                                            )

            summed = Assoc(row_union,
                           col_union,
                           1.0,
                           _sparse.add(Aadj_reindex, Badj_reindex),
                           aggregate="unique"
                           )
            summed.condense()
        else:
            # If one associative array is numerical (and the other is necessarily not), convert other via .logical()
            warning_message = (
                "When adding numerical and non-numerical associative arrays, the latter is converted"
                + "via .logical(). This may produce undesired results!"
            )
            self_ = self
            if (
                isinstance(self.val, float) and len(self.row) > 0
            ):  # Check numerical AND nonempty
                other = other.logical(copy=True)
                warnings.warn(warning_message)
            if isinstance(other.val, float) and len(other.row) > 0:
                self_ = self.logical(copy=True)
                warnings.warn(warning_message)

            summed = self_.combine(other, util.add)
        return summed

    def __sub__(self, other: "Assoc") -> "Assoc":
        """Take arithmetic difference of numerical associative arrays and set difference otherwise."""
        # If both associative arrays are numerical, compute the element-wise arithmetic difference.
        if isinstance(self.val, float) and isinstance(other.val, float):
            other_row, other_col, other_val = other.find()
            _other = Assoc(other_row, other_col, -other_val)

            return self + _other
        else:
            # Otherwise, delete from self any entries having a corresponding non-null entry in other
            def _minus(object_1, object_2):
                if object_1 not in Assoc.null_values and object_2 in Assoc.null_values:
                    return object_1
                else:
                    # Try to pick the 'correct' null value
                    if util.is_numeric(object_1):
                        return 0
                    elif isinstance(object_1, str):
                        return ""
                    else:
                        return None

            return self.combine(other, _minus, right_zero=True, left_zero=True)

    # Overload matrix multiplication
    def __matmul__(self, other: Union[int, float, "Assoc"]) -> "Assoc":
        """Array multiplication of A and B, with A's column indices matched up with B's row indices
        Usage:
            self @ other
        Input:
            other = Associative array
        Output:
            self @ other = array multiplication of self and other
        Note:
            - When either self or other are non-numerical the .logical() method is run on them.
        """
        self_ = self
        if not isinstance(self.val, float):
            self_ = self.logical(copy=True)

        if isinstance(other, Assoc):
            other_ = other
            if not isinstance(other.val, float):
                other_ = other.logical(copy=True)

            # Intersect A.col and B.row
            intersection, index_map_1, index_map_2 = util.sorted_intersect(
                self_.col, other_.row, return_index=True
            )

            # Get appropriate sub-matrices and multiply
            self_sparse = _sparse.getitem_cols(self_.adj, index_map_1)
            other_sparse = _sparse.getitem_rows(other_.adj, index_map_2)
            product_sparse = _sparse.matmul(self_sparse, other_sparse)
            product_sparse = _sparse.coo_canonicalize(product_sparse)

            product = Assoc(
                self_.row, other_.col, 1.0, product_sparse, aggregate="unique"
            )

            # Remove empty rows and columns
            product.condense()
            return product
        else:
            assert isinstance(other, float) or isinstance(other, int)
            if other == 0:
                return Assoc([], [], [])
            else:
                self_.adj = self_.adj.multiply(other)
                return self_

    def __rmatmul__(self, other):
        return self @ other

    def _assocmultiply(self, other):
        return self @ other

    # element-wise multiplication
    def __mul__(self, other: "Assoc") -> "Assoc":
        """Element-wise multiplication of self and B, matched up by row and column indices.
        Usage:
            self * other
        Input:
            other = Associative array
        Output:
            self * other = element-wise product of self and other if both are numerical, and otherwise
                self * other is the associative array whose triples are those triples
                (row_key, col_key, value) of self for which (row_key, col_key) corresponds with a nonempty
                entry in other.
        Note:
            - In the case where at least one of self and other are non-numerical, it may not be (and often
                isn't) the case that self * other equals other * self. I.e., when allowing
                non-numerical associative arrays, commutativity of (self, other) -> self * other is not
                guaranteed.
        """
        self_ = self.dropzeros(copy=True)
        other_ = other.dropzeros(copy=True)

        # Only multiply if both numerical, so logical() as appropriate
        if not (isinstance(self_.val, float) and isinstance(other_.val, float)):
            other_ = other_.logical(copy=False)
            warnings.warn(
                "If A or B are non-numerical, then A.multiply(B) returns the associative array with triples"
                + "from A whose corresponding entries in B are non-null."
                + "This may produce undesired results!"
            )

        row_int, row_index_self, row_index_other = util.sorted_intersect(
            self_.row, other_.row, return_index=True
        )
        col_int, col_index_self, col_index_other = util.sorted_intersect(
            self_.col, other_.col, return_index=True
        )

        self_sub = self_.adj.tocsr()[row_index_self, :][:, col_index_self]
        other_sub = other_.adj.tocsr()[row_index_other, :][:, col_index_other]
        multiplied_adj = self_sub.multiply(other_sub).tocoo()

        if isinstance(self_.val, float) and isinstance(other_.val, float):
            multiplied_val = 1.0
        else:
            multiplied_val = self_.val
            multiplied_adj = multiplied_adj.astype(int)

        multiplied = Assoc(
            row_int, col_int, multiplied_val, multiplied_adj, aggregate="unique"
        )
        multiplied.condense()
        multiplied.deepcondense()

        return multiplied

    def __rmul__(self, other):
        return self * other

    def multiply(self, other):
        return self * other

    # element-wise division -- for division by zero, treat as null
    def divide(self, other: Union["Assoc", Number]) -> "Assoc":
        """Element-wise division of self and B, matched up by row and column indices.
        Usage:
            self.divide(other)
        Input:
            other = Associative array or number
        Output:
            self.divide(other) = element-wise quotient of self by other
        Note:
            - Removes all explicit zeros and only takes division of non-null entries in self and other (if an
                associative array), effectively ignoring division by zero
            - Implicitly runs .logical() method on non-numerical associative arrays
        """
        if not isinstance(self.val, float):
            self_ = self.logical(copy=True)
        else:
            self_ = self.dropzeros(copy=True)

        if isinstance(other, Assoc):
            if not isinstance(other.val, float):
                other_inv = other.logical(copy=True)
            else:
                other_inv = other.dropzeros(copy=True)

            other_inv.adj.data = np.reciprocal(
                other_inv.adj.data.astype(float, copy=False)
            )

            return self_.multiply(other_inv)
        else:
            assert isinstance(other, float) or isinstance(other, int)
            try:
                other = 1 / other
                return self._assocmultiply(other)
            except ValueError:
                raise ValueError("Division by 0.")

    def min(
        self,
        other: Union["Assoc", KeyVal],
        sort_key: Optional[Callable[[KeyVal, KeyVal], bool]] = None,
    ) -> "Assoc":
        """Element-wise minimum between associative arrays. Supports optional comparison function."""
        if sort_key is None:
            comp_min = min
        else:

            def comp_min(x, y):
                try:
                    if sort_key(x, y):
                        return x
                    elif sort_key(y, x):
                        return y
                    elif x == y:
                        return y
                    else:
                        raise ValueError(
                            str(x)
                            + " and "
                            + str(y)
                            + " are incomparable with respect to sort_key "
                            + "function; minimum cannot be resolved."
                        )
                except TypeError:
                    raise TypeError(
                        "sort_key does not support comparison between "
                        + str(x)
                        + " and "
                        + str(y)
                        + "."
                    )

        return self.combine(other, comp_min, right_zero=True, left_zero=True)

    def max(
        self,
        other: Union["Assoc", KeyVal],
        sort_key: Optional[Callable[[KeyVal, KeyVal], bool]] = None,
    ) -> "Assoc":
        """Element-wise maximum between associative arrays. Supports optional comparison function."""
        if sort_key is None:
            comp_max = max
        else:

            def comp_max(x, y):
                try:
                    if sort_key(x, y):
                        return y
                    elif sort_key(y, x):
                        return x
                    elif x == y:
                        return y
                    else:
                        raise ValueError(
                            str(x)
                            + " and "
                            + str(y)
                            + " are incomparable with respect to sort_key "
                            + "function; maximum cannot be resolved."
                        )
                except TypeError:
                    raise TypeError(
                        "sort_key does not support comparison between "
                        + str(x)
                        + " and "
                        + str(y)
                        + "."
                    )

        return self.combine(other, comp_max, right_zero=True, left_zero=True)

    def __and__(self, other: "Assoc") -> "Assoc":
        """Element-wise logical AND of self and other, matched up by row and column indices.
        Usage:
            self & other
        Input:
            other = Associative array
        Output:
            self & other = element-wise logical AND (as a function {0,1} x {0,1} -> {0,1}) of self.logical() and
                other.logical()
        """
        # TODO: Should this take values of type bool?
        self_, other_ = self.logical(copy=True), other.logical(copy=True)
        return self_.multiply(other_)

    def __or__(self, other: "Assoc") -> "Assoc":
        """Element-wise logical OR (on {0,1}) of self and B, matched up by row and column indices.
        Usage:
            self | other
        Input:
            other = Associative array
        Output:
            self | other = element-wise logical OR (as a function {0,1} x {0,1} -> {0,1}) of self.logical() and
                other.logical()
        """
        # TODO: Should this take values of type bool?
        self_, other_ = self.logical(copy=True), other.logical(copy=True)
        return (self_ + other_).logical(copy=False)

    def sqin(self) -> "Assoc":
        """self.transpose() @ self"""
        return self.transpose()._assocmultiply(self)

    def sqout(self) -> "Assoc":
        """self @ self.transpose()"""
        return self._assocmultiply(self.transpose())

    def catkeymul(self, other: "Assoc", delimiter: str = ";") -> "Assoc":
        """Computes the array product, but values are delimiter-separated string list of
        the row/column indices which contribute to the value in the product
        Usage:
            self.catkeymul(other)
            self.catkeymul(other, delimiter)
        Input:
            other = Associative Array
            delimiter = (Optional, default ';') delimiter to separate the row/column indices
        Output:
            self.catkeymul(other) = Associative array where the (i,j)-th entry is null unless the (i,j)-th entry
                of self.logical() @ other.logical()  is not null, in which case that entry is the string list of
                the k-indices for which self[i,k] and other[k,j] were non-zero.
        """
        self_log, other_log = self.logical(), other.logical()
        C = self_log._assocmultiply(other_log)

        self_dict, other_dict = self_log.to_dict(), other_log.transpose().to_dict()
        rows = {
            row_key: np.unique(list(self_dict[row_key].keys()))
            for row_key in self_dict.keys()
        }
        cols = {
            col_key: np.unique(list(other_dict[col_key].keys()))
            for col_key in other_dict.keys()
        }

        catkey_row, catkey_col, _ = C.find()
        catkeys = list()

        for index in range(len(catkey_row)):
            row_key, col_key = catkey_row[index], catkey_col[index]
            common_keys = util.sorted_intersect(rows[row_key], cols[col_key]).astype(
                str
            )
            catkeys.append(delimiter.join(common_keys) + delimiter)

        return Assoc(catkey_row, catkey_col, catkeys)

    def catvalmul(
        self, other: "Assoc", pair_delimiter: str = ",", delimiter: str = ";"
    ) -> "Assoc":
        """Computes the array product, but values are delimiter-separated string list of
        the values of A and B which contribute to the value in the product
        Usage:
            self.catvalmul(other)
        Input:
            other = Associative Array
            pair_delimiter = (Optional, default ',') delimiter to separate the values in A and B
            delimiter = (Optional, default ';') delimiter to separate the value pairs
        Output:
            self.catvalmul(other) = Associative array where the (i,j)-th entry is null unless the (i,j)-th entry
                of self.logical() @ other.logical()  is not null, in which case that entry is the string list of
                the non-trivial value pairs 'self[i,k],other[k,j],'.
        """
        self_log, other_log = self.logical(copy=True), other.logical(copy=True)
        C = self_log._assocmultiply(other_log)

        self_dict, other_dict = self.to_dict(), other.transpose().to_dict()
        rows = {
            row_key: np.unique(list(self_dict[row_key].keys()))
            for row_key in self_dict.keys()
        }
        cols = {
            col_key: np.unique(list(other_dict[col_key].keys()))
            for col_key in other_dict.keys()
        }

        catval_row, catval_col, _ = C.find()
        catvals = list()

        for index in range(len(catval_row)):
            row_key, col_key = catval_row[index], catval_col[index]
            common_keys = util.sorted_intersect(rows[row_key], cols[col_key])
            catval = list()
            for common_key in common_keys:
                value_pair = [
                    self_dict[row_key][common_key],
                    other_dict[col_key][common_key],
                ]
                value_pair = util.num_to_str(value_pair)
                catval.append(pair_delimiter.join(value_pair) + pair_delimiter)
            catvals.append(delimiter.join(catval) + delimiter)

        return Assoc(catval_row, catval_col, catvals)

    def compare(
        self,
        other: Union["Assoc", KeyVal],
        sort_key: Callable[[KeyVal, KeyVal], bool],
        inverse: bool = False,
        include_inverse: bool = False,
    ) -> Union["Assoc", Tuple["Assoc", "Assoc"]]:
        """Generic element-wise comparison with another associative array or a single value according to sort_key.
        Usage:
            self.compare(other, min)
            self.compare(other, max, inverse=True)
        Inputs:
            other = Associative array or singular value
            inverse = (Optional, default False) Boolean indicating whether the sort_key function should be
                inverted, as if pre-composing the sort_key function with transposition
            include_inverse = (Optional, default False) Boolean indicating whether the result of
                self.compare(other, sort_key, inverse=True) should be included with
                self.compare(other, sort_key)
        Outputs:
            self.compare(other, sort_key) = Associative array whose triples are of the form
                (row_key, cold_key, 1) for (row_key, col_key) = (r, c) satisfying
                sort_key(self[r, c], other[r, c]) == True in the case where other is an associative array and
                self[r, c] or other[r, c] are possibly null (but not both), or
                sort_key(self[r, c], other) == True in the case where other is a singular value and self[r, c]
                is non-null.
            self.compare(other, sort_key, inverse=True) = As above, but with 'True' replaced by 'False'
            self.compare(other, sort_key, include_inverse=True) = the pair of self.compare(other, sort_key)
                and self.compare(other, sort_key, inverse=True)
        Note:
            - Comparisons are only made with explicitly stored entries of the associative array(s). E.g.,
                self.compare(1, <=) would only make comparisons between non-null entries of self with 1.
            - When comparisons are made between non-null values and null, '' is used for string values and 0 is
                used for numerical values.
            - If include_inverse=True, then the value of the Boolean inverse is ignored.
        """
        warnings.warn(
            "Comparisons are made only with explicitly stored entries of the associative array(s)."
        )

        inverse = False if include_inverse else inverse
        include_inverse = False if inverse else include_inverse

        self_dict = self.to_dict2()
        self_keys = set(self_dict.keys())
        if isinstance(other, Assoc):
            other_dict = other.to_dict2()
            other_keys = set(other_dict.keys())
            inter_keys = self_keys.intersection(other_keys)
            self_only_keys, other_only_keys = self_keys.difference(
                inter_keys
            ), other_keys.difference(self_keys)
        else:
            other_dict, other_keys = dict(), set()
            inter_keys, self_only_keys, other_only_keys = self_keys, set(), set()

        compared_row, compared_col = list(), list()
        inv_compared_row, inv_compared_col = list(), list()

        def _incompatible_message(x, y):
            return (
                "sort_key does not support comparison between "
                + str(x)
                + " and "
                + str(y)
                + "."
            )

        key_sets = [self_only_keys, inter_keys, other_only_keys]
        for index in range(3):
            for key in key_sets[index]:
                if index == 0:
                    self_value = self_dict[key]
                    other_value = 0 if util.is_numeric(self_value) else ""
                elif index == 1:
                    self_value = self_dict[key]
                    other_value = other_dict[key] if isinstance(other, Assoc) else other
                else:
                    other_value = other_dict[key]
                    self_value = 0 if util.is_numeric(other_value) else ""

                try:
                    comparison = (
                        sort_key(self_value, other_value)
                        if not inverse
                        else sort_key(other_value, self_value)
                    )
                    if comparison:
                        compared_row.append(key[0])
                        compared_col.append(key[1])
                except TypeError:
                    if inverse:
                        self_value, other_value = other_value, self_value
                    print(_incompatible_message(self_value, other_value))

                if include_inverse:
                    try:
                        if sort_key(other_value, self_value):
                            inv_compared_row.append(key[0])
                            inv_compared_col.append(key[1])
                    except TypeError:
                        print(_incompatible_message(other_value, self_value))

        if not include_inverse:
            return Assoc(compared_row, compared_col, 1)
        else:
            return Assoc(compared_row, compared_col, 1), Assoc(
                inv_compared_row, inv_compared_col, 1
            )

    def __eq__(self, other: Union["Assoc", KeyVal]) -> "Assoc":
        """Element-wise equality comparison between self and other.
        Usage:
            self == other
        Input:
            other = other object, e.g., another associative array, a number, or a string
        Output:
            self == other = An associative array such that for row and column labels r and c, resp., such that
                (self == other)(r,c) = 1 if and only if...
                    (Case 1) self(r,c) == other(r,c) (when other is another associative array
                        and assuming at least one of self(r,c) and other(r,c) is not null)
                    (Case 2) self(r,c) == other (when other is not another associative array)
                otherwise (self == other)(r,c) = null.
        Notes:
            - Only numeric and string data types are supported.
        Warnings:
            - Only compares values corresponding to keys explicitly stored in self or other.
        """

        def KeyVal_eq(value_1, value_2):
            if isinstance(value_1, str) and isinstance(value_2, str):
                return value_1 == value_2
            elif util.is_numeric(value_1) and util.is_numeric(value_2):
                return np.isclose(value_1, value_2)
            elif not (
                issubclass(type(value_1), type(value_2))
                or issubclass(type(value_2), type(value_1))
            ):
                return False
            else:
                raise TypeError

        return self.compare(other, sort_key=KeyVal_eq)

    def __ne__(self, other: Union["Assoc", KeyVal]) -> "Assoc":
        """Element-wise equality comparison between self and other.
        Usage:
            self != other
        Input:
            other = other object, e.g., another associative array, a number, or a string
        Output:
            self != other = An associative array such that for row and column labels r and c, resp., such that
                (self != other)(r,c) = 1 if and only if...
                    (Case 1) self(r,c) != other(r,c) (when other is another associative array
                        and assuming at least one of self(r,c) and other(r,c) is not null)
                    (Case 2) self(r,c) != other (when other is not another associative array)
                otherwise (self != other)(r,c) = null.
        Notes:
            - Only numeric and string data types are supported.
        Warnings:
            - Only compares values corresponding to keys explicitly stored in self or other.
        """

        def KeyVal_ne(value_1, value_2):
            if isinstance(value_1, str) and isinstance(value_2, str):
                return value_1 != value_2
            elif util.is_numeric(value_1) and util.is_numeric(value_2):
                return not np.isclose(value_1, value_2)
            elif not (
                issubclass(type(value_1), type(value_2))
                or issubclass(type(value_2), type(value_1))
            ):
                return True
            else:
                raise TypeError

        return self.compare(other, sort_key=KeyVal_ne)

    def __lt__(self, other: Union["Assoc", KeyVal]) -> "Assoc":
        """Element-wise equality comparison between self and other.
        Usage:
            self < other
        Input:
            other = other object, e.g., another associative array, a number, or a string
        Output:
            self < other = An associative array such that for row and column labels r and c, resp., such that
                (self < other)(r,c) = 1 if and only if...
                    (Case 1) self(r,c) < other(r,c) (when other is another associative array
                        and assuming at least one of self(r,c) and other(r,c) is not null)
                    (Case 2) self(r,c) < other (when other is not another associative array)
                otherwise (self < other)(r,c) = null.
        Notes:
            - Only numeric and string data types are supported.
        Warnings:
            - Only compares values corresponding to keys explicitly stored in self or other.
        """

        def KeyVal_lt(value_1, value_2):
            if isinstance(value_1, str) and isinstance(value_2, str):
                return value_1 < value_2
            elif util.is_numeric(value_1) and util.is_numeric(value_2):
                return not np.isclose(value_1, value_2) and value_1 < value_2
            elif not (
                issubclass(type(value_1), type(value_2))
                or issubclass(type(value_2), type(value_1))
            ):
                return False
            else:
                raise TypeError

        return self.compare(other, sort_key=KeyVal_lt)

    def __gt__(self, other: Union["Assoc", KeyVal]) -> "Assoc":
        """Element-wise equality comparison between self and other.
        Usage:
            self > other
        Input:
            other = other object, e.g., another associative array, a number, or a string
        Output:
            self > other = An associative array such that for row and column labels r and c, resp., such that
                (self > other)(r,c) = 1 if and only if...
                    (Case 1) self(r,c) > other(r,c) (when other is another associative array
                        and assuming at least one of self(r,c) and other(r,c) is not null)
                    (Case 2) self(r,c) > other (when other is not another associative array)
                otherwise (self > other)(r,c) = null.
        Notes:
            - Only numeric and string data types are supported.
        Warnings:
            - Only compares values corresponding to keys explicitly stored in self or other.
        """

        def KeyVal_gt(value_1, value_2):
            if isinstance(value_1, str) and isinstance(value_2, str):
                return value_1 > value_2
            elif util.is_numeric(value_1) and util.is_numeric(value_2):
                return not np.isclose(value_1, value_2) and value_1 > value_2
            elif not (
                issubclass(type(value_1), type(value_2))
                or issubclass(type(value_2), type(value_1))
            ):
                return False
            else:
                raise TypeError

        return self.compare(other, sort_key=KeyVal_gt)

    def __le__(self, other: Union["Assoc", KeyVal]) -> "Assoc":
        """Element-wise equality comparison between self and other.
        Usage:
            self <= other
        Input:
            other = other object, e.g., another associative array, a number, or a string
        Output:
            self <= other = An associative array such that for row and column labels r and c, resp., such that
                (self <= other)(r,c) = 1 if and only if...
                    (Case 1) self(r,c) <= other(r,c) (when other is another associative array
                        and assuming at least one of self(r,c) and other(r,c) is not null)
                    (Case 2) self(r,c) <= other (when other is not another associative array)
                otherwise (self <= other)(r,c) = null.
        Notes:
            - Only numeric and string data types are supported.
        Warnings:
            - Only compares values corresponding to keys explicitly stored in self or other.
        """

        def KeyVal_le(value_1, value_2):
            if isinstance(value_1, str) and isinstance(value_2, str):
                return value_1 <= value_2
            elif util.is_numeric(value_1) and util.is_numeric(value_2):
                return np.isclose(value_1, value_2) or value_1 <= value_2
            elif not (
                issubclass(type(value_1), type(value_2))
                or issubclass(type(value_2), type(value_1))
            ):
                return False
            else:
                raise TypeError

        return self.compare(other, sort_key=KeyVal_le)

    def __ge__(self, other: Union["Assoc", KeyVal]) -> "Assoc":
        """Element-wise equality comparison between self and other.
        Usage:
            self >= other
        Input:
            other = other object, e.g., another associative array, a number, or a string
        Output:
            self >= other = An associative array such that for row and column labels r and c, resp., such that
                (self >= other)(r,c) = 1 if and only if...
                    (Case 1) self(r,c) >= other(r,c) (when other is another associative array
                        and assuming at least one of self(r,c) and other(r,c) is not null)
                    (Case 2) self(r,c) >= other (when other is not another associative array)
                otherwise (self >= other)(r,c) = null.
        Notes:
            - Only numeric and string data types are supported.
        Warnings:
            - Only compares values corresponding to keys explicitly stored in self or other.
        """

        def KeyVal_ge(value_1, value_2):
            if isinstance(value_1, str) and isinstance(value_2, str):
                return value_1 >= value_2
            elif util.is_numeric(value_1) and util.is_numeric(value_2):
                return np.isclose(value_1, value_2) or value_1 >= value_2
            elif not (
                issubclass(type(value_1), type(value_2))
                or issubclass(type(value_2), type(value_1))
            ):
                return False
            else:
                raise TypeError

        return self.compare(other, sort_key=KeyVal_ge)


def nnz(A: "Assoc") -> int:
    return A.nnz()


def hadamard(A: "Assoc", B: "Assoc") -> "Assoc":
    return A * B


def combine(
    A: "Assoc",
    B: "Assoc",
    binary_op: Callable[[KeyVal, KeyVal], KeyVal],
    right_zero: bool = False,
    left_zero: bool = False,
) -> "Assoc":
    return A.combine(B, binary_op, right_zero=right_zero, left_zero=left_zero)


def assoc_min(
    A: Union[KeyVal, Assoc],
    B: Union[KeyVal, Assoc],
    sort_key: Optional[Callable[[KeyVal, KeyVal], bool]] = None,
) -> "Assoc":
    if isinstance(A, Assoc):
        return A.min(B, sort_key=sort_key)
    else:
        assert isinstance(B, Assoc)
        return B.min(A, sort_key=sort_key)


def assoc_max(
    A: Union[KeyVal, Assoc],
    B: Union[KeyVal, Assoc],
    sort_key: Optional[Callable[[KeyVal, KeyVal], bool]] = None,
) -> "Assoc":
    if isinstance(A, Assoc):
        return A.max(B, sort_key=sort_key)
    else:
        assert isinstance(B, Assoc)
        return B.max(A, sort_key=sort_key)


def transpose(A: "Assoc", copy: bool = True) -> "Assoc":
    return A.transpose(copy=copy)


def sqin(A: "Assoc") -> "Assoc":
    return A.sqin()


def sqout(A: "Assoc") -> "Assoc":
    return A.sqout()


def catkeymul(A: "Assoc", B: "Assoc", delimiter: str = ";") -> "Assoc":
    return A.catkeymul(B, delimiter=delimiter)


def catvalmul(
    A: "Assoc", B: "Assoc", pair_delimiter: str = ",", delimiter: str = ";"
) -> "Assoc":
    return A.catvalmul(B, pair_delimiter=pair_delimiter, delimiter=delimiter)


def val2col(A: "Assoc", separator: str = "|", int_aware: bool = True) -> "Assoc":
    """Convert from adjacency array to incidence array.
    Usage:
        val2col(A, separator)
    Inputs:
        A = Associative Array
        separator = (Optional, defualt '|') (new) delimiting character to separate column labels from values
        int_aware = (Optional, default True) Boolean which determines if '.0' is stripped from float strings
    Output:
        val2col(A, separator) = Associative Array B where B.row == A.row and
            B[row_label, col_label + split_separator + value] == 1 if and only if
            A[row_label, col_label] == value
    """
    rows, column_types, column_vals = A.find()
    column_types = util.num_to_str(column_types, int_aware=int_aware)
    column_vals = util.num_to_str(column_vals, int_aware=int_aware)
    cols = util.catstr(column_types, column_vals, separator, int_aware=int_aware)
    return Assoc(rows, cols, 1)


def col_to_type(A: "Assoc", separator: str = "|", convert: bool = True) -> "Assoc":
    """Split column keys of associative array and sorts first part as column key and second part as value.
    Inverse of val2col.
        Usage:
            B = col_to_type(A, separator)
        Inputs:
            A = Associative array with string column keys assumed to be of the form 'key' + separator + 'val'
            separator = (Optional, default '|') separator for A's column keys
            convert = (Optional, default True) Boolean indicating whether values should be considered numerical
                when possible
        Outputs:
            col_to_type(A, separator) = Associative array whose row keys are the same as A, but whose column
                keys are the first parts of A's column keys and whose values are the second parts of A's column keys
        Example:
            col_to_type(A, '|')
            col_to_type(A, '/')
        Note:
            - A's column keys must be in the desired form.
    """
    # Extract row and column keys from A
    row, col, _ = A.find()

    # Split column keys according to splitSep
    column_splits = list()
    try:
        column_splits = [column_key.split(separator) for column_key in col]
    except ValueError:
        print("Input column keys not of correct form.")

    # Extract column types and values
    try:
        column_types = [split_column_key[0] for split_column_key in column_splits]
    except IndexError:
        raise IndexError("Input column keys not of correct form.")
    try:
        column_vals = [split_column_key[1] for split_column_key in column_splits]
        if convert:
            # Convert only if every value is successfully converted to numerical
            try:
                column_vals = util.str_to_num(column_vals, silent=False)
            except ValueError:
                pass
    except IndexError:
        raise IndexError("Input column keys not of correct form.")

    return Assoc(row, column_types, column_vals)


col2type = col_to_type


def num_to_str(A: "Assoc", labels_only: bool = False, values_only: bool = False) -> "Assoc":
    new_row, new_col, new_val = A.find()

    if not values_only:
        new_row = new_row.astype(str)
        new_col = new_col.astype(str)
    if not labels_only:
        new_val = new_val.astype(str)

    B = Assoc(new_row, new_col, new_val)
    return B


num2str = num_to_str


def is_empty_assoc(A: "Assoc") -> bool:
    row_empty = (len(A.row) == 0)
    col_empty = (len(A.col) == 0)
    if isinstance(A.val, float):
        val_empty = (_sparse.nnz(A.adj) == 0)
    else:
        assert isinstance(A.val, np.ndarray)
        val_empty = (len(A.val) == 0)
    return row_empty or col_empty or val_empty


def assoc_equal(A: "Assoc", B: "Assoc", return_info: bool = False) -> bool:
    """Test whether two associative arrays are equal."""
    is_equal = True

    if not np.array_equal(A.row, B.row):
        is_equal = False
        if return_info:
            print(
                "Rows unequal:"
                + str(A.row)
                + " (dtype="
                + str(A.row.dtype)
                + ")"
                + " vs. "
                + str(B.row)
                + " (dtype="
                + str(B.row.dtype)
                + ")"
            )

    if not np.array_equal(A.col, B.col):
        is_equal = False
        if return_info:
            print(
                "Cols unequal:"
                + str(A.col)
                + " (dtype="
                + str(A.col.dtype)
                + ")"
                + " vs. "
                + str(B.col)
                + " (dtype="
                + str(B.col.dtype)
                + ")"
            )

    if not (
        (
            isinstance(A.val, float)
            and isinstance(B.val, float)
            and A.val == 1
            and B.val == 1
        )
        or np.array_equal(A.val, B.val)
    ):
        is_equal = False
        if return_info:
            dtype_1_message = ""
            if isinstance(A.val, np.ndarray):
                dtype_1_message += " (dtype=" + str(A.val.dtype) + ")"
            dtype_2_message = ""
            if isinstance(B.val, np.ndarray):
                dtype_2_message += " (dtype=" + str(B.val.dtype) + ")"
            print(
                "Vals unequal:"
                + str(A.val)
                + dtype_1_message
                + " vs. "
                + str(B.val)
                + dtype_2_message
            )

    if not _sparse.sparse_equal(A.adj, B.adj):
        is_equal = False
        if return_info:
            print("Adjs unequal:" + str(A.adj) + " vs. " + str(B.adj))

    return is_equal


def readcsvtotriples(
    filename: str,
    labels: bool = True,
    convert_keys: bool = False,
    convert_values: bool = False,
    convert: bool = False,
    triples: bool = False,
    **fmtoptions
) -> Tuple[List[KeyVal], List[KeyVal], List[KeyVal]]:
    """Read CSV file to row, col, val lists.
    Usage:
        row, col, val = readCSV(filename, labels=False, triples=False)
        row, col, val = readCSV(filename, fmtoptions)
    Inputs:
        filename = name of file (string)
        labels = (Optional, default True) Boolean indicating if row and column labels are the first column and row,
            resp.
        triples = (Optional, default False) Boolean indicating if each row is of the form 'row[i], col[i], val[i]'
        convert_keys = (Optional, default False) Boolean indicating if row and column keys should be converted
            from strings to numbers
        convert_values = (Optional, default False) Boolean indicating if values should be converted from strings to
            numbers
        convert = (Optional, default False) Boolean = conjunction of convert_keys and convert_values
        **fmtoptions = format options accepted by csv.reader, e.g., "delimiter='\t'" for tsv's
    Outputs:
        row, col, val = value in row[i]-th row and col[i]-th column is val[i] (if not triples,
            else the transposes of the columns)
    Examples:
        row, col, val = readcsv('my_file_name.csv')
        row, col, val = readcsv('my_file_name.csv', delimiter=';')
        row, col, val = readcsv('my_file_name.tsv', triples=True, delimiter='\t')
    """
    if convert:
        convert_values = True
        convert_keys = True

    # Read CSV file and create (row-index,col-index):value dictionary
    with open(filename, newline="") as csv_file:
        assoc_reader = csv.reader(csv_file, **fmtoptions)

        if triples:
            row, col, val = list(), list(), list()

            for line in assoc_reader:
                if len(line) == 0:
                    continue
                if len(line) != 3:
                    raise ValueError(
                        "line has "
                        + str(len(line))
                        + " elements:\n"
                        + str(line)
                        + "\ntriples=True implies there are three columns"
                    )
                else:
                    row.append(line[0])
                    col.append(line[1])
                    val.append(line[2])
        else:
            assoc_dict = dict()

            # If labels are expected, take first row to be the column labels
            if labels:
                headings = next(assoc_reader)
            else:
                line_num = 0  # Otherwise start counting the lines

            for row in assoc_reader:
                if len(row) == 0:
                    continue
                if labels and len(row) != len(headings):
                    raise ValueError(
                        "row has "
                        + str(len(row))
                        + " elements while there are "
                        + str(len(headings))
                        + " column labels."
                    )
                else:
                    # If labels are expected, first element of row is row label, otherwise actual value
                    if labels:
                        start = 1
                    else:
                        start = 0

                    for i in range(start, len(row)):
                        # If labels are expected, use with dictionary
                        if row[i] is not None and row[i] != "":
                            if labels:
                                assoc_dict[(row[0], headings[i])] = row[i]
                            else:
                                assoc_dict[(line_num, i)] = row[i]
                # Increment line counter
                if not labels:
                    line_num += 1

            # Extract row, col, val from dictionary
            row_col_tuples = list(assoc_dict.keys())
            row = [item[0] for item in row_col_tuples]
            col = [item[1] for item in row_col_tuples]
            val = list(assoc_dict.values())
            if convert_keys:
                try:
                    row = util.str_to_num(row)
                except ValueError:
                    pass
                try:
                    col = util.str_to_num(col)
                except ValueError:
                    pass
            if convert_values:
                try:
                    val = util.str_to_num(val)
                except ValueError:
                    pass

    return row, col, val


def readcsv(
    filename: str,
    labels: bool = True,
    triples: bool = False,
    convert_keys: bool = False,
    convert_values: bool = False,
    convert: bool = False,
    **fmtoptions
) -> "Assoc":
    """Read CSV file to Assoc instance.
    Usage:
        A = readcsv(filename)
        A = readcsv(filename, fmtoptions)
    Inputs:
        filename = name of file (string)
        labels = (Optional, default True) Boolean indicating if row and column labels are the first column and row,
            resp.
        triples = (Optional, default False) Boolean indicating if each row is of the form 'row[i], col[i], val[i]'
        convert_keys = (Optional, default False) Boolean indicating if row and column keys should be converted
            from strings to numbers
        convert_values = (Optional, default False) Boolean indicating if values should be converted from strings to
            numbers
        convert = (Optional, default False) Boolean = conjunction of convert_keys and convert_values
        fmtoptions = format options accepted by csv.reader, e.g. "delimiter='\t'" for tsv's
    Outputs:
        A = Associative Array whose column indices are given in the first line of the file, whose row indices
            are given in the first column of the file, and whose values are the remaining non-empty/null items,
            paired with the appropriate row and col indices
    Examples:
        A = readcsv('my_file_name.csv')
        A = readcsv('my_file_name.csv', delimiter=';')
        A = readcsv('my_file_name.tsv', delimiter='\t')
    """
    row, col, val = readcsvtotriples(
        filename,
        labels=labels,
        triples=triples,
        convert_keys=convert_keys,
        convert_values=convert_values,
        convert=convert,
        **fmtoptions
    )

    return Assoc(row, col, val)


def writecsv(A: "Assoc", filename: str, **fmtparams) -> None:
    """Write CSV file from Assoc instance.
    Usage:
        writeCSV(filename)
        writeCSV(filename, **fmtoptions)
    Inputs:
        A = Associative array to write to CSV
        filename = name of file to write to (string)
        fmtoptions = format options accepted by csv.writer
    Outputs:
        None
    Examples:
        writeCSV(A, 'my_file_name.csv')
        writeCSV(A, 'my_file_name.csv', delimiter=';')
    """
    with open(filename, "w") as csv_file:
        assoc_writer = csv.writer(csv_file, **fmtparams, lineterminator="\n")

        # Write the headings (offset by one to account for row indices)
        headings = [item for item in A.col]
        headings.insert(0, None)
        assoc_writer.writerow(headings)

        adj_dict = A.to_dict2()  # lookup dictionary

        for row_index in range(len(A.row)):
            new_line = list()
            new_line.append(A.row[row_index])

            for col_index in range(len(A.col)):
                if (A.row[row_index], A.col[col_index]) in adj_dict.keys():
                    new_line.append(adj_dict[(A.row[row_index], A.col[col_index])])
                else:
                    new_line.append(None)

            assoc_writer.writerow(new_line)

    return None


def read_mat(fname: str) -> Union[Dict[Any, "Assoc"], "Assoc"]:
    """Read .mat files created within Matlab/Octave D4M."""
    x = io.loadmat(fname)
    Assoc_dict = {}
    for key in x.keys():
        if hasattr(x[key], 'classname') and (getattr(x[key], 'classname') == 'Assoc'):
            Aout = x[key][0][0]
            row = Aout[0][0].split(Aout[0][0][-1])[0:-1]
            col = Aout[1][0].split(Aout[1][0][-1])[0:-1]
            val = Aout[2]
            if val.size == 0:
                val = 1.0
                Asparse = Aout[3].tocoo()
                Assoc_dict[key] = Assoc(row, col, val, Asparse)
            else:
                val = val[0].split(val[0][-1])[0:-1]
                Asparse = Aout[3]
                r, c, v = sparse.find(Asparse)
                row = [row[i] for i in r.astype(int)]
                col = [col[i] for i in c.astype(int)]
                val = [val[i-1] for i in v.astype(int)]
                Assoc_dict[key] = Assoc(row, col, val)

    if len(Assoc_dict.keys()) == 1:
        return Assoc_dict[list(Assoc_dict.keys())[0]]
    else:
        return Assoc_dict


def save_mat(fname: str, A: "Assoc", name: str = "Aout", **spio_savemat_kwargs) -> NotImplemented:
    """Create .mat file compatible with Matlab/Octave D4M."""
    mat_dict = {}
    row, col, val = A.find()
    row_cat = "\t".join(row.astype(str))
    col_cat = "\t".join(col.astype(str))
    adj = A.adj.tocsc()

    if isinstance(A.val, float):
        arr = np.array([[row_cat], [col_cat], [], adj])
    else:
        val_cat = "\t".join(row.astype(str))
        arr = np.array([[row_cat], [col_cat], [val_cat], adj], dtype=object)

    mat_dict[name] = arr

    io.savemat(fname, mat_dict, **spio_savemat_kwargs)
    return NotImplemented
