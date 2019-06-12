from D4M.assoc import *
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
    assert is_numeric(test) == exp


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
        union, index_map_1, index_map_2 = sorted_union(test1, test2, return_index=True)
        assert np.array_equal(union, exp)
        assert np.array_equal(index_map_1, returnexp1)
        assert np.array_equal(index_map_2, returnexp2)
    else:
        union = sorted_union(test1, test2)
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
        intersection, index_map_1, index_map_2 = sorted_intersect(test1, test2, return_index=True)
        assert np.array_equal(intersection, exp)
        assert np.array_equal(index_map_1, returnexp1)
        assert np.array_equal(index_map_2, returnexp2)
    else:
        intersection = sorted_intersect(test1, test2)
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
    assert np.array_equal(exp, sanitize(test, convert))


@pytest.mark.parametrize("test_row,test_col,test_val,exp_row,exp_col,exp_val,exp_adj",
                         [('a,b,', 'A,B,', 1, np.array(['a', 'b']), np.array(['A', 'B']), 1.0,
                           np.array([[1.0, 0], [0, 1.0]])),
                          ('a,b,b,', 'A,B,A,', 1, np.array(['a', 'b']), np.array(['A', 'B']), 1.0,
                           np.array([[1.0, 0], [1.0, 1.0]]))
                          ])
def test_assoc_constructor(test_row, test_col, test_val, exp_row, exp_col, exp_val, exp_adj):
    A = Assoc(test_row, test_col, test_val)
    assert np.array_equal(A.row, exp_row)
    assert np.array_equal(A.col, exp_col)
    assert np.array_equal(A.val, exp_val) or (A.val == 1.0 and exp_val == 1.0)
    assert np.array_equal(A.adj.toarray(), exp_adj)
