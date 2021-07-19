import D4M.assoc
import pytest
import numpy as np
import scipy.sparse as sp


@pytest.mark.parametrize("test,exp",
                         [(1.0, True),
                          (-1, True),
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


@pytest.mark.parametrize("test,testsub,exp",
                         [("aa,bb,ab,", "a,", [0, 2]),
                          ("aa,bb,ab,", "b,", [1, 2]),
                          ("aa,bb,ab,", "c,", []),
                          ("aa,bb,ab,", "a,b,", [0, 1, 2]),
                          (['aa', 'bb', 'ab'], "a,", [0, 2]),
                          (['aa', 'bb', 'ab'], "b,", [1, 2]),
                          (['aa', 'bb', 'ab'], "c,", []),
                          (['aa', 'bb', 'ab'], "a,b,", [0, 1, 2]),
                          (['aa', 'bb', 'ab'], ['a'], [0, 2]),
                          (['aa', 'bb', 'ab'], ['b'], [1, 2]),
                          (['aa', 'bb', 'ab'], ['c'], []),
                          (['aa', 'bb', 'ab'], ['a', 'b'], [0, 1, 2]),
                          ("aa,bb,ab,", ['a'], [0, 2]),
                          ("aa,bb,ab,", ['b'], [1, 2]),
                          ("aa,bb,ab,", ['c'], []),
                          ("aa,bb,ab,", ['a', 'b'], [0, 1, 2])]
                         )
def test_contains(test, testsub, exp):
    assert D4M.assoc.contains(testsub)(test) == exp


@pytest.mark.parametrize("test,testsub,exp",
                         [("aa,bb,ab,", "a,", [0, 2]),
                          ("aa,bb,ab,", "b,", [1]),
                          ("aa,bb,ab,", "c,", []),
                          ("aa,bb,ab,", "a,b,", [0, 1, 2]),
                          (['aa', 'bb', 'ab'], "a,", [0, 2]),
                          (['aa', 'bb', 'ab'], "b,", [1]),
                          (['aa', 'bb', 'ab'], "c,", []),
                          (['aa', 'bb', 'ab'], "a,b,", [0, 1, 2]),
                          (['aa', 'bb', 'ab'], ['a'], [0, 2]),
                          (['aa', 'bb', 'ab'], ['b'], [1]),
                          (['aa', 'bb', 'ab'], ['c'], []),
                          (['aa', 'bb', 'ab'], ['a', 'b'], [0, 1, 2]),
                          ("aa,bb,ab,", ['a'], [0, 2]),
                          ("aa,bb,ab,", ['b'], [1]),
                          ("aa,bb,ab,", ['c'], []),
                          ("aa,bb,ab,", ['a', 'b'], [0, 1, 2])]
                         )
def test_startswith(test, testsub, exp):
    assert D4M.assoc.startswith(testsub)(test) == exp


@pytest.mark.parametrize("obj,exp",
                         [("1", 1),
                          ("1.2", 1.2),
                          ("-5", -5)
                          ])
def test_str_to_num(obj, exp):
    assert D4M.assoc.str_to_num(obj) == exp


@pytest.mark.parametrize("obj,exp",
                         [([0, 1], ['0', '1']),
                          ([0, -1], ['0', '-1']),
                          ([0, 1, 0.12, -1], ['0.0', '1.0', '0.12', '-1.0'])
                          ])
def test_num_to_str(obj, exp):
    arr = np.array(obj)
    strarr = np.array(exp)
    assert np.array_equal(D4M.assoc.num_to_str(arr), strarr)


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


@pytest.mark.parametrize("row,col,val,func,agg_row,agg_col,agg_val",
                         [(['a', 'a', 'b'], ['A', 'A', 'B'], [1, 2, 3], D4M.assoc.add, ['a', 'b'], ['A', 'B'], [3, 3]),
                          (['a', 'a', 'b'], ['A', 'A', 'B'], [1, 2, 3], D4M.assoc.first,
                           ['a', 'b'], ['A', 'B'], [1, 3]),
                          (['a', 'a', 'b'], ['A', 'A', 'B'], [1, 2, 3], D4M.assoc.last, ['a', 'b'], ['A', 'B'], [2, 3]),
                          (['a', 'a', 'b'], ['A', 'A', 'B'], [2, 2, 3],
                           D4M.assoc.times, ['a', 'b'], ['A', 'B'], [4, 3]),
                          (['a', 'a', 'a', 'b'], ['A', 'A', 'A', 'B'], [1, 2, 0, 3], min, ['a', 'b'],
                           ['A', 'B'], [0, 3])
                          ])
def test_aggregate(row, col, val, func, agg_row, agg_col, agg_val):
    agg_row = np.array(agg_row)
    agg_col = np.array(agg_col)
    agg_val = np.array(agg_val)

    new_row, new_col, new_val = D4M.assoc.aggregate(row, col, val, func)
    assert np.array_equal(new_row, agg_row)
    assert np.array_equal(new_col, agg_col)
    assert np.array_equal(new_val, agg_val)


# @pytest.mark.parametrize("iterable,return_index,return_inverse,unique,index_map,index_map_inverse",
#                          [([9, 5, 7, 1, 0, 1, 5, 6], True, True, [0, 1, 5, 6, 7, 9],
#                            [4, 3, 1, 7, 2, 0], [5, 2, 4, 1, 0, 1, 2, 3]),
#                           ([9, 5, 7, 1, 0, 1, 5, 6], False, False, [0, 1, 5, 6, 7, 9],
#                            None, None),
#                           ([9, 5, 7, 1, 0, 1, 5, 6], True, False, [0, 1, 5, 6, 7, 9],
#                            [4, 3, 1, 7, 2, 0], None),
#                           ([9, 5, 7, 1, 0, 1, 5, 6], False, True, [0, 1, 5, 6, 7, 9],
#                            None, [5, 2, 4, 1, 0, 1, 2, 3]),
#                           (np.array([9, 5, 7, 1, 0, 1, 5, 6]), True, True, np.array([0, 1, 5, 6, 7, 9]),
#                            np.array([4, 3, 1, 7, 2, 0]), np.array([5, 2, 4, 1, 0, 1, 2, 3])),
#                           (np.array([9, 5, 7, 1, 0, 1, 5, 6]), False, False, np.array([0, 1, 5, 6, 7, 9]),
#                            None, None),
#                           (np.array([9, 5, 7, 1, 0, 1, 5, 6]), True, False, np.array([0, 1, 5, 6, 7, 9]),
#                            np.array([4, 3, 1, 7, 2, 0]), None),
#                           (np.array([9, 5, 7, 1, 0, 1, 5, 6]), False, True, np.array([0, 1, 5, 6, 7, 9]),
#                            None, np.array([5, 2, 4, 1, 0, 1, 2, 3])),
#                           (['abra', 'kadabra', 'alakazam', 'machop', 'machoke', 'machamp'], True, True,
#                            ['abra', 'alakazam', 'kadabra', 'machamp', 'machoke', 'machop'],
#                            [0, 2, 1, 5, 4, 3], [0, 2, 1, 5, 4, 3])
#                           ])
# def test_unique(iterable, return_index, return_inverse, unique, index_map, index_map_inverse):
#     result = D4M.assoc.unique(iterable, return_index=return_index, return_inverse=return_inverse)
#
#     if isinstance(iterable, np.ndarray):
#
#
#         if return_index or return_inverse:
#             assert np.array_equal(unique, result)
#         else:
#             assert np.array_equal(unique, result[0])
#
#             if return_index:
#                 assert index_map == result[1]
#
#                 if return_inverse:
#                     assert index_map_inverse == result[2]
#                 else:
#                     assert index_map_inverse is None
#             else:
#                 assert index_map is None
#
#                 if return_inverse:
#                     assert index_map_inverse == result[1]
#                 else:
#                     assert index_map_inverse is None
#     else:
#         exp = tuple([item for item in [unique, index_map, index_map_inverse] if item is not None])
#         print(exp)
#         print(D4M.assoc.unique(iterable, return_index=return_index, return_inverse=return_inverse))
#
#         assert (exp == D4M.assoc.unique(iterable, return_index=return_index, return_inverse=return_inverse))
#         assert not return_index == index_map is None
#         assert not return_inverse == index_map_inverse is None


@pytest.mark.parametrize("s1,s2,sep,exp",
                         [(np.array(['a', 'b']), np.array(['A', 'B']), None, np.array(['a|A', 'b|B'])),
                          (np.array(['a', 'b']), np.array(['A', 'B']), ':', np.array(['a:A', 'b:B'])),
                          (np.array([1, 1]), np.array(['A', 'B']), None, np.array(['1|A', '1|B']))
                          ])
def test_catstr(s1, s2, sep, exp):
    assert np.array_equal(exp, D4M.assoc.catstr(s1, s2, sep))


"""
@pytest.mark.parametrize("row,col,val,splitSep,exp",
                         [('a,b,a,', 'A,B,B,', [1, 2, 3], None, D4M.assoc.Assoc('a,b,a,', 'A|1,B|2,B|3,', 1)),
                          ('a,b,a,', 'A,B,B,', 'aA,bB,aB,', None, D4M.assoc.Assoc('a,b,a,', 'A|aA,B|bB,B|aB,', 1)),
                          ('a,b,a,', 'A,B,B,', [1, 2, 3], ':', D4M.assoc.Assoc('a,b,a,', 'A:1,B:2,B:3,', 1))
                          ])
def test_val2col(row, col, val, splitSep, exp):
    A = D4M.assoc.Assoc(row, col, val)
    assert array_equal(D4M.assoc.val2col(A, splitSep), exp)
"""


# @pytest.mark.parametrize("A,splitSep,row,col,val",
#                          [(D4M.assoc.Assoc('a,b,a,', 'A|1,B|2,B|3,', 1), None, 'a,b,a,', 'A,B,B,', [1, 2, 3]),
#                           (D4M.assoc.Assoc('a,b,a,', 'A|aA,B|bB,B|aB,', 1), None, 'a,b,a,', 'A,B,B,', 'aA,bB,aB,'),
#                           (D4M.assoc.Assoc('a,b,a,', 'A:1,B:2,B:3,', 1), ':', 'a,b,a,', 'A,B,B,', [1, 2, 3])
#                           ])
# def test_col2type(A, splitSep, row, col, val):
#     B = D4M.assoc.Assoc(row, col, val)
#     assert array_equal(D4M.assoc.col2type(A, splitSep), B)


def sparse_equal(A: sp.spmatrix, B: sp.spmatrix):
    """ Test whether two COO sparse matrices are equal. """
    # isequal = False
    #
    # #eqdtype = A.dtype == B.dtype
    # eqshape = A.shape == B.shape
    # eqndim = A.ndim == B.ndim
    # eqnnz = A.nnz == B.nnz
    # eqdata = np.array_equal(A.data, B.data)
    # eqrow = np.array_equal(A.row, B.row)
    # eqcol = np.array_equal(A.col, B.col)
    #
    # if eqshape and eqndim and eqnnz and eqdata and eqrow and eqcol:
    #     isequal = True
    return (A != B).nnz == 0


def array_equal(A, B):
    """ Test whether two associative arrays are equal. """
    is_equal = True

    if np.array_equal(A.row, B.row):
        pass
    else:
        is_equal = False
        print("Rows unequal:"+str(A.row)+" vs. "+str(B.row))

    if np.array_equal(A.col, B.col):
        pass
    else:
        is_equal = False
        print("Cols unequal:" + str(A.col) + " vs. " + str(B.col))

    if (isinstance(A.val, float) and isinstance(B.val, float) and A.val == 1 and B.val == 1)\
            or np.array_equal(A.val, B.val):
        pass
    else:
        is_equal = False
        print("Vals unequal:" + str(A.val) + " vs. " + str(B.val))

    if sparse_equal(A.adj, B.adj):
        pass
    else:
        is_equal = False
        print("Adjs unequal:" + str(A.adj) + " vs. " + str(B.adj))

    return is_equal


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


# @pytest.mark.parametrize("test_row,test_col,test_val,test_adj",
#                          [('a,b,c,', 'A,B,C,', 1.0, sp.coo_matrix(np.array([[1.0, 3.0], [0, 2.0]]))),
#                           ])
# def test_assoc_constructor_sparse_bad_dims(test_row, test_col, test_val, test_adj):
#     with pytest.raises(Exception) as e_info:
#         D4M.assoc.Assoc(test_row, test_col, test_val, test_adj)
#     assert str(e_info.value) == "Unique row and column indices do not match sp_matrix."


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


@pytest.mark.parametrize("test_assoc,ordering,row,col,val",
                         [(D4M.assoc.Assoc('a,b,', 'A,B,', [2, 3]), None, np.array(['a', 'b']),
                           np.array(['A', 'B']), np.array([2, 3])),
                          (D4M.assoc.Assoc('a,b,a,b,', 'A,B,B,A,', 'aA,bB,aB,bA,'), None,
                           np.array(['a', 'b', 'a', 'b']), np.array(['A', 'B', 'B', 'A']),
                           np.array(['aA', 'bB', 'aB', 'bA'])),
                          (D4M.assoc.Assoc('a,b,a,b,', 'A,B,B,A,', 'aA,bB,aB,bA,'), 0,
                           np.array(['a', 'a', 'b', 'b']), np.array(['A', 'B', 'A', 'B']),
                           np.array(['aA', 'aB', 'bA', 'bB'])),
                          (D4M.assoc.Assoc('a,b,a,b,', 'A,B,B,A,', 'aA,bB,aB,bA,'), 1,
                           np.array(['a', 'b', 'a', 'b']), np.array(['A', 'A', 'B', 'B']),
                           np.array(['aA', 'bA', 'aB', 'bB']))
                          ])
def test_find(test_assoc, ordering, row, col, val):
    test_row, test_col, test_val = test_assoc.find(ordering=ordering)
    if ordering is None:
        triples = list(zip(list(row), list(col), list(val)))
        triples.sort()
        print(triples)
        test_triples = list(zip(list(test_row), list(test_col), list(test_val)))
        test_triples.sort()
        print(test_triples)
        assert triples == test_triples
    else:
        assert np.array_equal(test_row, row)
        assert np.array_equal(test_col, col)
        assert np.array_equal(test_val, val)


@pytest.mark.parametrize("test_assoc,ordering,triples",
                         [(D4M.assoc.Assoc('a,b,', 'A,B,', [2, 3]), None, [('a', 'A', 2), ('b', 'B', 3)]),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', 'aA,bB,'), None, [('a', 'A', 'aA'), ('b', 'B', 'bB')]),
                          (D4M.assoc.Assoc('a,b,a,b,', 'A,B,B,A,', 'aA,bB,aB,bA,'), None,
                           [('a', 'A', 'aA'), ('b', 'B', 'bB'), ('a', 'B', 'aB'), ('b', 'A', 'bA')]),
                          (D4M.assoc.Assoc('a,b,a,b,', 'A,B,B,A,', 'aA,bB,aB,bA,'), 0,
                           [('a', 'A', 'aA'), ('a', 'B', 'aB'), ('b', 'A', 'bA'), ('b', 'B', 'bB')]),
                          (D4M.assoc.Assoc('a,b,a,b,', 'A,B,B,A,', 'aA,bB,aB,bA,'), 1,
                           [('a', 'A', 'aA'), ('b', 'A', 'bA'), ('a', 'B', 'aB'), ('b', 'B', 'bB')])
                          ])
def test_assoc_triples(test_assoc, ordering, triples):
    if ordering is None:
        test_triples = test_assoc.triples()
        test_triples.sort()
        list_triples = list(triples)
        list_triples.sort()
        assert test_triples == list_triples
    else:
        assert test_assoc.triples(ordering=ordering) == triples



@pytest.mark.parametrize("test_assoc,rowkey,colkey,value",
                         [(D4M.assoc.Assoc('a,b,', 'A,B,', [2, 3]), 'b', 'B', 3),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,color,color,', 'amber,auburn,white,'),
                           'amber', 'color', 'amber'),
                          (D4M.assoc.Assoc('a,b,a,b,', 'A,B,B,A,', 'aA,bB,aB,bA,'),
                           'a', 'B', 'aB'),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', [2, 3]), 'c', 'B', 0)
                          ])
def test_getvalue(test_assoc, rowkey, colkey, value):
    assert test_assoc.getvalue(rowkey, colkey) == value


@pytest.mark.parametrize("assoc1,assoc2,equal",
                         [(D4M.assoc.Assoc('a,b,', 'A,B,', 1), D4M.assoc.Assoc('a,b,', 'A,B,', [1, 1]), True),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', 1), D4M.assoc.Assoc('a,b,', 'A,B,', [1.0, 1.0]), True),
                          (D4M.assoc.Assoc(['a', 'b'], ['A', 'B'], 1), D4M.assoc.Assoc('a,b,', 'A,B,', 1), True),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', 1), D4M.assoc.Assoc('a,b,a,', 'A,B,B,', [1, 1, 1]), False),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', [1, 2]), D4M.assoc.Assoc('a,b,c,', 'A,B,C,', [1, 2, 0]),
                           True),
                          (D4M.assoc.Assoc('b,a,', 'B,A,', 1), D4M.assoc.Assoc('a,b,', 'A,B,', 1), True),
                          (D4M.assoc.Assoc('a,b,a,', 'A,B,B,', [1, 1, 0]).condense(),
                           D4M.assoc.Assoc('a,b,', 'A,B,', 1), True)
                          ])
def test_array_equal(assoc1, assoc2, equal):
    assert array_equal(assoc1, assoc2) == equal


@pytest.mark.parametrize("test_assoc,subrow,subcol,exp_assoc",
                         [(D4M.assoc.Assoc('a,b,', 'A,B,', [2, 3]), 'a,', 'A,B,', D4M.assoc.Assoc('a,', 'A,', [2])),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', 'aA,bB,'), 'a,', 'A,B,', D4M.assoc.Assoc('a,', 'A,', 'aA,')),
                          (D4M.assoc.Assoc('a,b,a,b,', 'A,B,B,A,', 'aA,bB,aB,bA,'), 'a,b,', 'A,',
                           D4M.assoc.Assoc('a,b,', 'A,A,', 'aA,bA,')),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,color,color,', 'amber,auburn,white,'),
                           D4M.assoc.startswith('a,'), ':',
                           D4M.assoc.Assoc('amber,ash,', 'color,color,', 'amber,auburn,')),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,height,leaf_color,', ['amber', 1.7, 'green']),
                           ":", D4M.assoc.contains('color,'),
                           D4M.assoc.Assoc('amber,birch,', 'color,leaf_color,', ['amber', 'green'])),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,color,color,', 'amber,auburn,white,'),
                           [0, 1], ":", D4M.assoc.Assoc('amber,ash,', 'color,color,', 'amber,auburn,')),
                          (D4M.assoc.Assoc('amber,ash,', 'color,color,', 'amber,auburn,'),
                           D4M.assoc.startswith('c,'), ":",
                           D4M.assoc.Assoc([], [], [])),
                          (D4M.assoc.Assoc('amber,ash,', 'color,color,', 'amber,auburn,'),
                           0, ":", D4M.assoc.Assoc('amber,', 'color,', 'amber,'))
                          ])
def test_getitem(test_assoc, subrow, subcol, exp_assoc):
    test_assoc.printfull()
    print(subrow)
    print(subcol)
    B = test_assoc[subrow, subcol]
    assert array_equal(B, exp_assoc)


@pytest.mark.parametrize("test_assoc",
                         [(D4M.assoc.Assoc('a,b,', 'A,B,', [2, 3])),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', 'aA,bB,')),
                          (D4M.assoc.Assoc('a,b,a,b,', 'A,B,B,A,', 'aA,bB,aB,bA,')),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,color,color,', 'amber,auburn,white,')),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,height,leaf_color,', ['amber', 1.7, 'green'])),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,color,color,', 'amber,auburn,white,')),
                          (D4M.assoc.Assoc('amber,ash,', 'color,color,', 'amber,auburn,')),
                          (D4M.assoc.Assoc('amber,ash,', 'color,color,', 'amber,auburn,'))
                          ])
def test_copy(test_assoc):
    exp_assoc = test_assoc.copy()

    assert array_equal(test_assoc, exp_assoc)

    assert not (test_assoc.row is exp_assoc.row)
    assert not (test_assoc.col is exp_assoc.col)
    if isinstance(test_assoc.val, float) or isinstance(exp_assoc.val, float):
        assert test_assoc.val == exp_assoc.val
    else:
        assert not (test_assoc.val is exp_assoc.val)


@pytest.mark.parametrize("test_assoc,rownum,colnum",
                         [(D4M.assoc.Assoc('a,b,', 'A,B,', [2, 3]), 2, 2),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', 'aA,bB,'), 2, 2),
                          (D4M.assoc.Assoc('a,b,a,b,', 'A,B,B,A,', 'aA,bB,aB,bA,'), 2, 2),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,color,color,', 'amber,auburn,white,'), 3, 1),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,height,leaf_color,', ['amber', 1.7, 'green']),
                           3, 3),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,color,color,', 'amber,auburn,white,'), 3, 1),
                          (D4M.assoc.Assoc('amber,ash,', 'color,color,', 'amber,auburn,'), 2, 1),
                          (D4M.assoc.Assoc([], [], []), 0, 0)
                          ])
def test_size(test_assoc, rownum, colnum):
    test_rownum, test_colnum = test_assoc.size()
    assert test_rownum == rownum
    assert test_colnum == colnum


@pytest.mark.parametrize("test_assoc,exp_nnz",
                         [(D4M.assoc.Assoc('a,b,', 'A,B,', [2, 0]), 1),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', 2), 2),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', 'aA,bB,'), 2),
                          (D4M.assoc.Assoc('a,b,a,b,', 'A,B,B,A,', 'aA,bB,aB,bA,'), 4),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,color,color,', 'amber,auburn,white,'), 3),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,height,leaf_color,', ['amber', 1.7, 'green']),
                           3),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,color,color,', 'amber,,white,'), 2),
                          (D4M.assoc.Assoc('amber,ash,', 'color,color,', 'amber,auburn,'), 2),
                          (D4M.assoc.Assoc('a,', 'A,', [None]), 0),
                          (D4M.assoc.Assoc([], [], []), 0)
                          ])
def test_nnz(test_assoc, exp_nnz):
    assert test_assoc.nnz() == exp_nnz

@pytest.mark.parametrize("test_assoc,exp_nonzeros",
                         [(D4M.assoc.Assoc('a,b,', 'A,B,', [2, 0]), 1),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', 2), 2),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', 'aA,bB,'), 2),
                          (D4M.assoc.Assoc('a,b,a,b,', 'A,B,B,A,', 'aA,bB,aB,bA,'), 4),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,color,color,', 'amber,auburn,white,'), 3),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,height,leaf_color,', ['amber', 1.7, 'green']),
                           3),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,color,color,', 'amber,,white,'), 2),
                          (D4M.assoc.Assoc('amber,ash,', 'color,color,', 'amber,auburn,'), 2),
                          (D4M.assoc.Assoc('a,', 'A,', [None]), 0),
                          (D4M.assoc.Assoc([], [], []), 0)
                          ])
def test_count_nonzero(test_assoc, exp_nonzeros):
    assert test_assoc.count_nonzero() == exp_nonzeros


@pytest.mark.parametrize("test_assoc,exp_assoc",
                         [(D4M.assoc.Assoc('a,b,', 'A,B,', [2, 0]), D4M.assoc.Assoc('a,', 'A,', [2])),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', [0, 2]), D4M.assoc.Assoc('b,', 'B,', [2])),
                          (D4M.assoc.Assoc('a,b,c,', 'A,B,C,', [2, 0, 2]), D4M.assoc.Assoc('a,c,', 'A,C,', 2)),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', 2), D4M.assoc.Assoc('a,b,', 'A,B,', 2)),
                          (D4M.assoc.Assoc('a,b,', 'A,B,', 'aA,bB,'), D4M.assoc.Assoc('a,b,', 'A,B,', 'aA,bB,')),
                          (D4M.assoc.Assoc('amber,ash,birch,', 'color,color,color,', 'amber,,white,'),
                           D4M.assoc.Assoc('amber,birch,', 'color,color,', 'amber,white,')),
                          (D4M.assoc.Assoc('a,', 'A,', [None]), D4M.assoc.Assoc([], [], []))
                          ])
def test_dropzeros(test_assoc, exp_assoc):
    new_assoc = test_assoc.dropzeros(copy=True)
    test_assoc.printfull()
    new_assoc.printfull()
    exp_assoc.printfull()
    assert array_equal(new_assoc, exp_assoc)

