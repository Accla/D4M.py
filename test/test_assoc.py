import D4M.assoc
import pytest
import numpy as np
import scipy.sparse as sp


@pytest.mark.parametrize("test,exp",
                         [(1.0, True),
                          ('a', False),
                          ('1', False),
                          ('a1', False)
                          ])
def test_is_numeric(test, exp):
    assert D4M.assoc.is_numeric(test) == exp


@pytest.mark.parametrize("test1,test2,returnbool,exp,returnexp1,returnexp2",
                        [(np.array([0, 1, 4, 6]), np.array([0, 4, 7]),
                          True, np.array([0, 1, 4, 6, 7]),
                          np.array([0, 1, 2, 3]), np.array([0, 2, 4])),
                         (np.array([0, 1, 4, 6]), np.array([0, 4, 7]),
                          False, np.array([0, 1, 4, 6, 7]),
                          None, None)
                         ])
def test_sorted_union(test1, test2, returnbool, exp, returnexp1, returnexp2):
    if returnbool:
        union, index_map_1, index_map_2 = D4M.assoc.sorted_union(test1, test2, return_index=True)
        assert np.array_equal(union, exp)
        assert np.array_equal(index_map_1, returnexp1)
        assert np.array_equal(index_map_2, returnexp2)
    else:
        union = D4M.assoc.sorted_union(test1, test2)
        assert np.array_equal(union, exp)


@pytest.mark.parametrize("test1,test2,returnbool,exp,returnexp1,returnexp2",
                         [(np.array([0, 1, 4]), np.array([0, 4, 7]),
                           True, np.array([0, 4]),
                           np.array([0, 2]), np.array([0, 1])),
                          (np.array([0, 1, 4]), np.array([0, 4, 7]),
                           False, np.array([0, 4]),
                           None, None)
                          ])
def test_sorted_intersect(test1, test2, returnbool, exp, returnexp1, returnexp2):
    if returnbool:
        intersection, index_map_1, index_map_2 = D4M.assoc.sorted_intersect(test1, test2, return_index=True)
        assert np.array_equal(intersection, exp)
        assert np.array_equal(index_map_1, returnexp1)
        assert np.array_equal(index_map_2, returnexp2)
    else:
        intersection = D4M.assoc.sorted_intersect(test1, test2)
        assert np.array_equal(intersection, exp)


@pytest.mark.parametrize("test,convert,exp",
                         [([1, 1], False, np.array([1, 1], dtype=object)),
                          (1, False, np.array([1], dtype=object)),
                          ('a,b,', False, np.array(['a', 'b'], dtype=object)),
                          ('1,1,', False, np.array(['1', '1'], dtype=object)),
                          ('1,1,', True, np.array([1, 1], dtype=object)),
                          ('a,b,', True, np.array(['a', 'b'], dtype=object))
                          ])
def test_sanitize(test, convert, exp):
    assert np.array_equal(exp, D4M.assoc.sanitize(test, convert))


def sparse_equal(A, B):
    """ Test whether two COO sparse matrices are equal."""
    isequal = False

    eqdtype = A.dtype == B.dtype
    eqshape = A.shape == B.shape
    eqndim = A.ndim == B.ndim
    eqnnz = A.nnz == B.nnz
    eqdata = np.array_equal(A.data, B.data)
    eqrow = np.array_equal(A.row, B.row)
    eqcol = np.array_equal(A.col, B.col)

    if eqdtype and eqshape and eqndim and eqnnz and eqdata and eqrow and eqcol:
        isequal = True

    return isequal


@pytest.mark.parametrize("test_row,test_col,test_val,arg,exp_row,exp_col,exp_val,exp_adj",
                         [('a,b,', 'A,B,', 1, None, np.array(['a', 'b']), np.array(['A', 'B']), 1.0,
                           sp.coo_matrix(np.array([[1.0, 0], [0, 1.0]]))),
                          ('a,b,b,', 'A,B,A,', 1, None, np.array(['a', 'b']), np.array(['A', 'B']), 1.0,
                           sp.coo_matrix(np.array([[1.0, 0], [1.0, 1.0]]))),
                          ('a,b,', 'A,B,', 'aA,bB,', None, np.array(['a', 'b']), np.array(['A', 'B']),
                           np.array(['aA', 'bB']), sp.coo_matrix(np.array([[1, 0], [0, 2]]))),
                          (['a', 'b'], 'A,B,', [1, 1], None, np.array(['a', 'b']), np.array(['A', 'B']), 1.0,
                           sp.coo_matrix(np.array([[1.0, 0], [0, 1.0]]))),
                          ([1, 2], 'A,B,', [1, 1], None, np.array([1, 2]), np.array(['A', 'B']), 1.0,
                           sp.coo_matrix(np.array([[1.0, 0], [0, 1.0]]))),
                          ([], 'A,B,', 1, None, np.empty(0), np.empty(0), 1.0,
                           sp.coo_matrix(([], ([], [])), shape=(0, 0))),
                          ('a,b,a,', 'A,B,A,', [1, 2, 3], D4M.assoc.add, np.array(['a', 'b']), np.array(['A', 'B']),
                           1.0, sp.coo_matrix(np.array([[4.0, 0], [0, 2.0]]))),
                          ('a,b,a,', 'A,B,A,', [1, 2, 3], D4M.assoc.first, np.array(['a', 'b']), np.array(['A', 'B']),
                           1.0, sp.coo_matrix(np.array([[1.0, 0], [0, 2.0]]))),
                          ('a,b,a,', 'A,B,A,', [1, 2, 3], D4M.assoc.last, np.array(['a', 'b']), np.array(['A', 'B']),
                           1.0, sp.coo_matrix(np.array([[3.0, 0], [0, 2.0]]))),
                          ('a,b,a,', 'A,B,A,', [1, 2, 3], min, np.array(['a', 'b']), np.array(['A', 'B']),
                           1.0, sp.coo_matrix(np.array([[1.0, 0], [0, 2.0]]))),
                          ('a,b,a,', 'A,B,A,', [1, 2, 3], max, np.array(['a', 'b']), np.array(['A', 'B']),
                           1.0, sp.coo_matrix(np.array([[3.0, 0], [0, 2.0]]))),
                          ('a,b,a,', 'A,B,A,', [3, 2, 3], D4M.assoc.times, np.array(['a', 'b']), np.array(['A', 'B']),
                           1.0, sp.coo_matrix(np.array([[9.0, 0], [0, 2.0]]))),
                          ('a,b,a,', 'A,B,B,', 1.0, sp.coo_matrix(np.array([[1.0, 3.0], [0, 2.0]])),
                           np.array(['a', 'b']), np.array(['A', 'B']), 1.0,
                           sp.coo_matrix(np.array([[1.0, 3.0], [0, 2.0]]))),
                          ('a,b,a,', 'A,B,B,', ['aA', 'bA', 'aB'], sp.coo_matrix(np.array([[1, 3], [0, 2]])),
                           np.array(['a', 'b']), np.array(['A', 'B']), np.array(['aA', 'aB', 'bA']),
                           sp.coo_matrix(np.array([[1, 3], [0, 2]]))),
                          ('a,b,', 'A,B,', ['aA', 'bA', 'aB'], sp.coo_matrix(np.array([[1, 3], [0, 2]])),
                           np.array(['a', 'b']), np.array(['A', 'B']), np.array(['aA', 'aB', 'bA']),
                           sp.coo_matrix(np.array([[1, 3], [0, 2]]))),
                          ])
def test_assoc_constructor(test_row, test_col, test_val, arg, exp_row, exp_col, exp_val, exp_adj):
    A = D4M.assoc.Assoc(test_row, test_col, test_val, arg)
    assert np.array_equal(A.row, exp_row)
    assert np.array_equal(A.col, exp_col)
    assert np.array_equal(A.val, exp_val) or (A.val == 1.0 and exp_val == 1.0)
    assert sparse_equal(A.adj, exp_adj)


@pytest.mark.parametrize("test_row,test_col,test_val",
                         [('a,b,c,', 'A,B,', 1),
                          (['a', 'b', 'c'], 'A,B,', 1),
                          ])
def test_assoc_constructor_incompatible_lengths(test_row, test_col, test_val):
    with pytest.raises(Exception) as e_info:
        D4M.assoc.Assoc(test_row, test_col, test_val)
    assert str(e_info.value) == 'Invalid input: row, col, val must have compatible lengths.'


@pytest.mark.parametrize("test_row,test_col,test_val,test_adj,info",
                         [('a,', 'A,B,B,', ['aA', 'bA', 'aB'], sp.coo_matrix(np.array([[1, 3], [0, 2]])),
                           "Invalid input: not enough unique row indices."),
                          ('a,', 'A,', ['aA', 'bA', 'aB'], sp.coo_matrix(np.array([[1, 3], [0, 2]])),
                           "Invalid input: not enough unique row indices, not enough unique col indices."),
                          ('a,b,', 'A,B,', ['aA'], sp.coo_matrix(np.array([[1, 3], [0, 2]])),
                           "Invalid input: not enough unique values."),
                          ])
def test_assoc_constructor_sparse_too_small(test_row, test_col, test_val, test_adj, info):
    with pytest.raises(Exception) as e_info:
        D4M.assoc.Assoc(test_row, test_col, test_val, test_adj)
    assert str(e_info.value) == info


@pytest.mark.parametrize("test_row,test_col,test_val,test_adj",
                         [('a,b,c,', 'A,B,C,', 1.0, sp.coo_matrix(np.array([[1.0, 3.0], [0, 2.0]]))),
                          ])
def test_assoc_constructor_sparse_bad_dims(test_row, test_col, test_val, test_adj):
    with pytest.raises(Exception) as e_info:
        D4M.assoc.Assoc(test_row, test_col, test_val, test_adj)
    assert str(e_info.value) == "Unique row and column indices do not match sp_matrix."

@pytest.mark.parametrize("test_row,test_col,test_val,arg,val",
                         [('a,b,', 'A,B,', 1, None, np.array([1.0])),
                          ('a,b,b,', 'A,B,A,', 1, None, np.array([1.0])),
                          ('a,b,', 'A,B,', 'aA,bB,', None, np.array(['aA', 'bB'])),
                          (['a', 'b'], 'A,B,', [1, 1], None, np.array([1])),
                          ([1, 2], 'A,B,', [1, 1], None, np.array([1])),
                          ([], 'A,B,', 1, None, np.empty(0)),
                          ('a,b,a,', 'A,B,A,', [1, 2, 3], D4M.assoc.add, np.array([2.0, 4.0])),
                          ('a,b,a,', 'A,B,A,', [1, 2, 3], D4M.assoc.first, np.array([1.0, 2.0])),
                          ('a,b,a,', 'A,B,A,', [1, 2, 3], D4M.assoc.last, np.array([2.0, 3.0])),
                          ('a,b,a,', 'A,B,A,', [1, 2, 3], min, np.array([1.0, 2.0])),
                          ('a,b,a,', 'A,B,A,', [1, 2, 3], max, np.array([2.0, 3.0])),
                          ('a,b,a,', 'A,B,A,', [3, 2, 3], D4M.assoc.times, np.array([2.0, 9.0])),
                          ('a,b,a,', 'A,B,B,', 1.0, sp.coo_matrix(np.array([[1.0, 3.0], [0, 2.0]])),
                           np.array([1.0, 2.0, 3.0])),
                          ('a,b,a,', 'A,B,B,', ['aA', 'bA', 'aB'], sp.coo_matrix(np.array([[1, 3], [0, 2]])),
                           np.array(['aA', 'aB', 'bA'])),
                          ('a,b,', 'A,B,', ['aA', 'bA', 'aB'], sp.coo_matrix(np.array([[1, 3], [0, 2]])),
                           np.array(['aA', 'aB', 'bA'])),
                          ])
def test_getval(test_row, test_col, test_val, arg, val):
    A = D4M.assoc.Assoc(test_row, test_col, test_val, arg)
    assert np.array_equal(val, A.getval())
