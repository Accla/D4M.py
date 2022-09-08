import pytest
import numpy as np
import scipy.sparse as sp
from numbers import Number

import D4M.assoc
import D4M.util
from D4M.util import _replace_default_args


@pytest.mark.parametrize(
    "test_row,test_col,test_val,adj,aggregate,exp_row,exp_col,exp_val,exp_adj",
    [
        (
            "a,b,",
            "A,B,",
            1,
            None,
            None,
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[1.0, 0], [0, 1.0]])),
        ),
        (
            "a,b,b,",
            "A,B,A,",
            1,
            None,
            None,
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[1.0, 0], [1.0, 1.0]])),
        ),
        (
            "a,b,",
            "A,B,",
            "aA,bB,",
            None,
            None,
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["aA", "bB"]),
            sp.coo_matrix(np.array([[1, 0], [0, 2]])),
        ),
        (
            ["a", "b"],
            "A,B,",
            [1, 1],
            None,
            None,
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[1.0, 0], [0, 1.0]])),
        ),
        (
            [1, 2],
            "A,B,",
            [1, 1],
            None,
            None,
            np.array([1, 2]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[1.0, 0], [0, 1.0]])),
        ),
        (
            [],
            "A,B,",
            1,
            None,
            None,
            np.empty(0),
            np.empty(0),
            1.0,
            sp.coo_matrix(([], ([], [])), shape=(0, 0)),
        ),
        (
            "a,b,a,",
            "A,B,A,",
            [1, 2, 3],
            None,
            D4M.util.add,
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[4.0, 0], [0, 2.0]])),
        ),
        (
            "a,b,a,",
            "A,B,A,",
            [1, 2, 3],
            None,
            "add",
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[4.0, 0], [0, 2.0]])),
        ),
        (
            "a,b,a,",
            "A,B,A,",
            [1, 2, 3],
            None,
            D4M.util.first,
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[1.0, 0], [0, 2.0]])),
        ),
        (
            "a,b,a,",
            "A,B,A,",
            [1, 2, 3],
            None,
            "first",
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[1.0, 0], [0, 2.0]])),
        ),
        (
            "a,b,a,",
            "A,B,A,",
            [1, 2, 3],
            None,
            D4M.util.last,
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[3.0, 0], [0, 2.0]])),
        ),
        (
            "a,b,a,",
            "A,B,A,",
            [1, 2, 3],
            None,
            "last",
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[3.0, 0], [0, 2.0]])),
        ),
        (
            "a,b,a,",
            "A,B,A,",
            [1, 2, 3],
            None,
            min,
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[1.0, 0], [0, 2.0]])),
        ),
        (
            "a,b,a,",
            "A,B,A,",
            [1, 2, 3],
            None,
            max,
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[3.0, 0], [0, 2.0]])),
        ),
        (
            "a,b,a,",
            "A,B,A,",
            [3, 2, 3],
            None,
            D4M.util.times,
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[9.0, 0], [0, 2.0]])),
        ),
        (
            "a,b,a,",
            "A,B,B,",
            1.0,
            sp.coo_matrix(np.array([[1.0, 3.0], [0, 2.0]])),
            None,
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[1.0, 3.0], [0, 2.0]])),
        ),
        (
            "a,b,a,",
            "A,B,B,",
            ["aA", "bA", "aB"],
            sp.coo_matrix(np.array([[1, 3], [0, 2]])),
            None,
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["aA", "aB", "bA"]),
            sp.coo_matrix(np.array([[1, 3], [0, 2]])),
        ),
        (
            "a,b,",
            "A,B,",
            ["aA", "bA", "aB"],
            sp.coo_matrix(np.array([[1, 3], [0, 2]])),
            None,
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["aA", "aB", "bA"]),
            sp.coo_matrix(np.array([[1, 3], [0, 2]])),
        ),
        (
            "a,b,",
            "A,B,",
            "aA,aB,bA,bB,",
            sp.coo_matrix(np.array([[1, 3], [4, 2]])),
            "unique",
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["aA", "aB", "bA", "bB"]),
            sp.coo_matrix(np.array([[1, 3], [4, 2]])),
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["aA", "aB", "bA", "bB"]),
            sp.coo_matrix(np.array([[1, 3], [4, 2]])),
            "unique",
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["aA", "aB", "bA", "bB"]),
            sp.coo_matrix(np.array([[1, 3], [4, 2]])),
        ),
        (
            np.array(["a", "b"]),
            np.array(["B", "A"]),
            np.array(["aA", "aB", "bA", "bB"]),
            sp.coo_matrix(np.array([[1, 3], [4, 2]])),
            "unique",
            np.array(["a", "b"]),
            np.array(["B", "A"]),
            np.array(["aA", "aB", "bA", "bB"]),
            sp.coo_matrix(np.array([[1, 3], [4, 2]])),
        ),
        (
            "a,b,",
            "B,A,",
            "aA,aB,bA,bB,",
            sp.coo_matrix(np.array([[1, 3], [4, 2]])),
            "unique",
            np.array(["a", "b"]),
            np.array(["B", "A"]),
            np.array(["aA", "aB", "bA", "bB"]),
            sp.coo_matrix(np.array([[1, 3], [4, 2]])),
        ),
    ],
)
def test_assoc_constructor(
    test_row, test_col, test_val, adj, aggregate, exp_row, exp_col, exp_val, exp_adj
):
    adj, aggregate = _replace_default_args(D4M.assoc.Assoc, adj=adj, aggregate=aggregate)
    assoc_ = D4M.assoc.Assoc(test_row, test_col, test_val, adj=adj, aggregate=aggregate)
    assert np.array_equal(assoc_.row, exp_row)
    assert np.array_equal(assoc_.col, exp_col)
    assert np.array_equal(assoc_.val, exp_val) or (assoc_.val == 1.0 and exp_val == 1.0)
    assert D4M.assoc.sparse_equal(assoc_.adj, exp_adj)


@pytest.mark.parametrize(
    "test_row,test_col,test_val,adj,aggregate,prevent_upcasting,convert,exp_row,exp_col,exp_val,"
    + "exp_adj",
    [
        (
            "a,b,c,",
            "A,A,B,",
            "1,2,3,",
            None,
            None,
            None,
            True,
            np.array(["a", "b", "c"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[1, 0], [2, 0], [0, 3]]), dtype=float),
        ),
        (
            "a,b,c,",
            "A,A,B,",
            "1,2,3,",
            None,
            None,
            True,
            False,
            np.array(["a", "b", "c"]),
            np.array(["A", "B"]),
            np.array(["1", "2", "3"]),
            sp.coo_matrix(np.array([[1, 0], [2, 0], [0, 3]]), dtype=int),
        ),
        (
            "a,b,c,",
            "A,A,B,",
            "1.0,2,3,",
            None,
            None,
            True,
            True,
            np.array(["a", "b", "c"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[1, 0], [2, 0], [0, 3]]), dtype=float),
        ),
        (
            "a,b,c,",
            "A,A,B,",
            "1,2,3,",
            None,
            None,
            True,
            True,
            np.array(["a", "b", "c"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(np.array([[1, 0], [2, 0], [0, 3]]), dtype=int),
        ),
    ],
)
def test_assoc_constructor_convert_upcast(
    test_row,
    test_col,
    test_val,
    adj,
    aggregate,
    prevent_upcasting,
    convert,
    exp_row,
    exp_col,
    exp_val,
    exp_adj,
):
    adj, aggregate = _replace_default_args(D4M.assoc.Assoc, adj=adj, aggregate=aggregate)
    assoc_ = D4M.assoc.Assoc(
        test_row,
        test_col,
        test_val,
        adj=adj,
        aggregate=aggregate,
        prevent_upcasting=prevent_upcasting,
        convert_val=convert,
    )
    assert np.array_equal(assoc_.row, exp_row)
    assert np.array_equal(assoc_.col, exp_col)
    assert np.array_equal(assoc_.val, exp_val) or (assoc_.val == 1.0 and exp_val == 1.0)
    assert D4M.assoc.sparse_equal(assoc_.adj, exp_adj)


@pytest.mark.parametrize(
    "test_row,test_col,test_val",
    [
        ("a,b,c,", "A,B,", 1),
        (["a", "b", "c"], "A,B,", 1),
    ],
)
def test_assoc_constructor_incompatible_lengths(test_row, test_col, test_val):
    with pytest.raises(Exception) as e_info:
        D4M.assoc.Assoc(test_row, test_col, test_val)
    assert (
        str(e_info.value)
        == "Invalid input: row, col, val must have compatible lengths."
    )


@pytest.mark.parametrize(
    "test_row,test_col,test_val,test_adj,info",
    [
        (
            "a,",
            "A,B,B,",
            ["aA", "bA", "aB"],
            sp.coo_matrix(np.array([[1, 3], [0, 2]])),
            "Invalid input: not enough unique row indices.",
        ),
        (
            "a,",
            "A,",
            ["aA", "bA", "aB"],
            sp.coo_matrix(np.array([[1, 3], [0, 2]])),
            "Invalid input: not enough unique row indices, not enough unique col indices.",
        ),
        (
            "a,b,",
            "A,B,",
            ["aA"],
            sp.coo_matrix(np.array([[1, 3], [0, 2]])),
            "Invalid input: not enough unique values.",
        ),
    ],
)
def test_assoc_constructor_sparse_too_small(
    test_row, test_col, test_val, test_adj, info
):
    with pytest.raises(Exception) as e_info:
        D4M.assoc.Assoc(test_row, test_col, test_val, test_adj)
    assert str(e_info.value) == info


@pytest.mark.parametrize(
    "test_row,test_col,test_val,test_adj,canonical",
    [
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(([1, 1, 1], ([0, 1, 1], [0, 0, 1])), dtype=float),
            True,
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["aA", "bB"]),
            sp.coo_matrix(([1, 2, 1], ([0, 1, 1], [0, 0, 1])), dtype=int),
            True,
        ),
        (
            ["a", "b"],
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(([1, 1, 1], ([0, 1, 1], [0, 0, 1])), dtype=float),
            False,
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            2.0,
            sp.coo_matrix(([1, 1, 1], ([0, 1, 1], [0, 0, 1])), dtype=float),
            False,
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(([1, 1, 1], ([0, 1, 1], [0, 0, 1])), dtype=float).tocsr(),
            False,
        ),
        (
            np.array(["a", "b"]),
            np.array(["B", "A"]),
            1.0,
            sp.coo_matrix(([1, 1, 1], ([0, 1, 1], [0, 0, 1])), dtype=float),
            False,
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B", "B"]),
            1.0,
            sp.coo_matrix(([1, 1, 1], ([0, 1, 1], [0, 0, 1])), dtype=float),
            False,
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(
                ([1, 1, 1], ([0, 1, 1], [0, 0, 1])), dtype=float, shape=(3, 3)
            ),
            False,
        ),
        (
            np.array(["a", "b", "c"]),
            np.array(["A", "B", "C"]),
            1.0,
            sp.coo_matrix(([1, 1, 1], ([0, 2, 2], [0, 0, 2])), dtype=float),
            False,
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(([1, 0, 1], ([0, 1, 1], [0, 0, 1])), dtype=float),
            False,
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["", "aA", "bB"]),
            sp.coo_matrix(([1, 2, 3], ([0, 1, 1], [0, 0, 1])), dtype=int),
            False,
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["bB", "aA"]),
            sp.coo_matrix(([1, 2, 1], ([0, 1, 1], [0, 0, 1])), dtype=int),
            False,
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["aA", "aA", "bB"]),
            sp.coo_matrix(([1, 2, 1], ([0, 1, 1], [0, 0, 1])), dtype=int),
            False,
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["aA", "bB"]),
            sp.coo_matrix(([1, 2, 0], ([0, 1, 1], [0, 0, 1])), dtype=int),
            False,
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["aA", "bB"]),
            sp.coo_matrix(([1, 2, 3], ([0, 1, 1], [0, 0, 1])), dtype=int),
            False,
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["aA", "bB"]),
            sp.coo_matrix(([1, 2, 1], ([0, 1, 1], [0, 0, 1])), dtype=float),
            False,
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["aA", "bA", "bB"]),
            sp.coo_matrix(([1, 3, 1], ([0, 1, 1], [0, 0, 1])), dtype=int),
            False,
        ),
    ],
)
def test_is_canonical(test_row, test_col, test_val, test_adj, canonical):
    test_assoc = D4M.assoc.Assoc(
        [], [], []
    )  # Modify empty array to ensure no modification of arguments
    test_assoc.row, test_assoc.col, test_assoc.val, test_assoc.adj = (
        test_row,
        test_col,
        test_val,
        test_adj,
    )
    assert test_assoc.is_canonical() == canonical


@pytest.mark.parametrize(
    "test_row,test_col,test_val,test_adj,exp_assoc",
    [
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(([2, 0], ([0, 1], [0, 1]))),
            D4M.assoc.Assoc("a,", "A,", [2]),
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(([0, 2], ([0, 1], [0, 1]))),
            D4M.assoc.Assoc("b,", "B,", [2]),
        ),
        (
            np.array(["a", "b", "c"]),
            np.array(["A", "B", "C"]),
            1.0,
            sp.coo_matrix(([2, 0, 2], ([0, 1, 2], [0, 1, 2]))),
            D4M.assoc.Assoc("a,c,", "A,C,", [2, 2]),
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(([2, 2], ([0, 1], [0, 1]))),
            D4M.assoc.Assoc("a,b,", "A,B,", 2),
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["aA", "bB"]),
            sp.coo_matrix(([1, 2], ([0, 1], [0, 1]))),
            D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,"),
        ),
        (
            np.array(["a", "b", "c"]),
            np.array(["A", "B", "C"]),
            np.array(["", "aA", "bB"]),
            sp.coo_matrix(([2, 3, 1], ([0, 1, 2], [0, 1, 2]))),
            D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,"),
        ),
        (
            np.array(["amber", "ash", "birch"]),
            np.array(["color"]),
            np.array(["", "amber", "white"]),
            sp.coo_matrix(([2, 1, 3], ([0, 1, 2], [0, 0, 0]))),
            D4M.assoc.Assoc("amber,birch,", "color,color,", "amber,white,"),
        ),
        (
            np.array(["a"]),
            np.array(["A"]),
            np.array([""]),
            sp.coo_matrix(([1], ([0], [0]))),
            D4M.assoc.Assoc([], [], []),
        ),
    ],
)
def test_dropzeros(test_row, test_col, test_val, test_adj, exp_assoc):
    test_assoc = D4M.assoc.Assoc(
        test_row, test_col, test_val, test_adj, aggregate="unique"
    )
    print(str(test_assoc))
    test_assoc.printfull()
    exp_assoc.printfull()
    new_assoc = test_assoc.dropzeros(copy=True)
    test_assoc.printfull()
    new_assoc.printfull()
    exp_assoc.printfull()
    assert D4M.assoc.assoc_equal(new_assoc, exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_row,test_col,test_val,test_adj,exp_assoc",
    [
        (
            np.array(["a", "b", "c"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix((np.array([1, 1]), (np.array([0, 1]), np.array([0, 1])))),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B", "C"]),
            1.0,
            sp.coo_matrix((np.array([1, 1]), (np.array([0, 1]), np.array([0, 1])))),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
        ),
        (
            np.array(["a", "b", "c"]),
            np.array(["A", "B", "C"]),
            np.array(["aA", "bB"]),
            sp.coo_matrix(
                (np.array([1, 2], dtype=int), (np.array([0, 1]), np.array([0, 1])))
            ),
            D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,"),
        ),
        (
            np.array(["a"]),
            np.array(["A"]),
            np.array([]),
            sp.coo_matrix((np.array([]), (np.array([]), np.array([]))), shape=(0, 0)),
            D4M.assoc.Assoc([], [], []),
        ),
    ],
)
def test_condense(test_row, test_col, test_val, test_adj, exp_assoc):
    test_assoc = D4M.assoc.Assoc(
        test_row, test_col, test_val, test_adj, aggregate="unique"
    )
    assert D4M.assoc.assoc_equal(test_assoc.condense(), exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_row,test_col,test_val,test_adj,exp_assoc",
    [
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            1.0,
            sp.coo_matrix(
                (np.array([1, 1, 1]), (np.array([0, 0, 1]), np.array([0, 1, 1])))
            ),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
        ),
        (
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array(["aA", "aB", "bB"]),
            sp.coo_matrix(
                (np.array([1, 3]), (np.array([0, 1]), np.array([0, 1]))), dtype=int
            ),
            D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,"),
        ),
        (
            np.array([]),
            np.array([]),
            np.array(["aA"]),
            sp.coo_matrix((np.array([]), (np.array([]), np.array([]))), shape=(0, 0)),
            D4M.assoc.Assoc([], [], []),
        ),
    ],
)
def test_deepcondense(test_row, test_col, test_val, test_adj, exp_assoc):
    test_assoc = D4M.assoc.Assoc(
        test_row, test_col, test_val, test_adj, aggregate="unique"
    )
    assert D4M.assoc.assoc_equal(test_assoc.deepcondense(), exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc,new_row,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            "1,2,",
            D4M.assoc.Assoc("1,1,2,", "A,B,B,", 1),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            "1,2,3,",
            D4M.assoc.Assoc("1,1,2,", "A,B,B,", 1),
        ),
    ],
)
def test_set_row(test_assoc, new_row, exp_assoc):
    assert D4M.assoc.assoc_equal(test_assoc.set_row(new_row), exp_assoc)


@pytest.mark.parametrize(
    "test_assoc,new_col,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            "1,2,",
            D4M.assoc.Assoc("a,a,b,", "1,2,2,", 1),
        )
    ],
)
def test_set_col(test_assoc, new_col, exp_assoc):
    assert D4M.assoc.assoc_equal(test_assoc.set_col(new_col), exp_assoc)


@pytest.mark.parametrize(
    "test_assoc,new_val,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            "1,2,",
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "1,"),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            "2,",
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "2,"),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            2,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 2),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA,aB,bB,"),
            "Aa,Ba,Bb,",
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "Aa,Ba,Bb,"),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA,aB,bB,"),
            0,
            D4M.assoc.Assoc([], [], []),
        ),
    ],
)
def test_set_val(test_assoc, new_val, exp_assoc):
    assert D4M.assoc.assoc_equal(
        test_assoc.set_val(new_val), exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "test_assoc,new_adj,numerical,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            sp.coo_matrix(([1, 2, 3], ([0, 0, 1], [0, 1, 1]))),
            None,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            sp.coo_matrix(([1], ([0], [0]))),
            None,
            D4M.assoc.Assoc("a,", "A,", 1),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA,aB,bB,"),
            sp.coo_matrix(([1, 2, 3], ([0, 0, 1], [0, 1, 1]))),
            None,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA,aB,bB,"),
            sp.coo_matrix(([1, 2, 3], ([0, 0, 1], [0, 1, 1]))),
            False,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA,aB,bB,"),
        ),
    ],
)
def test_set_adj(test_assoc, new_adj, numerical, exp_assoc):
    numerical = _replace_default_args(D4M.assoc.Assoc.set_adj, numerical=numerical)
    assert D4M.assoc.assoc_equal(
        test_assoc.set_adj(new_adj, numerical=numerical), exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "test_assoc,row_key,col_key,value,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            "c",
            "C",
            1.5,
            D4M.assoc.Assoc("a,a,b,c,", "A,B,B,C,", [1, 1, 1, 1.5]),
        )
    ],
)
def test_setitem(test_assoc, row_key, col_key, value, exp_assoc):
    test_assoc[row_key, col_key] = value
    assert D4M.assoc.assoc_equal(test_assoc, exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc,ordering,row,col,val",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", [2, 3]),
            None,
            np.array(["a", "b"]),
            np.array(["A", "B"]),
            np.array([2, 3]),
        ),
        (
            D4M.assoc.Assoc("a,b,a,b,", "A,B,B,A,", "aA,bB,aB,bA,"),
            None,
            np.array(["a", "b", "a", "b"]),
            np.array(["A", "B", "B", "A"]),
            np.array(["aA", "bB", "aB", "bA"]),
        ),
        (
            D4M.assoc.Assoc("a,b,a,b,", "A,B,B,A,", "aA,bB,aB,bA,"),
            0,
            np.array(["a", "a", "b", "b"]),
            np.array(["A", "B", "A", "B"]),
            np.array(["aA", "aB", "bA", "bB"]),
        ),
        (
            D4M.assoc.Assoc("a,b,a,b,", "A,B,B,A,", "aA,bB,aB,bA,"),
            1,
            np.array(["a", "b", "a", "b"]),
            np.array(["A", "A", "B", "B"]),
            np.array(["aA", "bA", "aB", "bB"]),
        ),
    ],
)
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


@pytest.mark.parametrize(
    "test_assoc,ordering,triples",
    [
        (D4M.assoc.Assoc("a,b,", "A,B,", [2, 3]), None, [("a", "A", 2), ("b", "B", 3)]),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,"),
            None,
            [("a", "A", "aA"), ("b", "B", "bB")],
        ),
        (
            D4M.assoc.Assoc("a,b,a,b,", "A,B,B,A,", "aA,bB,aB,bA,"),
            None,
            [("a", "A", "aA"), ("b", "B", "bB"), ("a", "B", "aB"), ("b", "A", "bA")],
        ),
        (
            D4M.assoc.Assoc("a,b,a,b,", "A,B,B,A,", "aA,bB,aB,bA,"),
            0,
            [("a", "A", "aA"), ("a", "B", "aB"), ("b", "A", "bA"), ("b", "B", "bB")],
        ),
        (
            D4M.assoc.Assoc("a,b,a,b,", "A,B,B,A,", "aA,bB,aB,bA,"),
            1,
            [("a", "A", "aA"), ("b", "A", "bA"), ("a", "B", "aB"), ("b", "B", "bB")],
        ),
        (D4M.assoc.Assoc([], [], []), None, []),
    ],
)
def test_assoc_triples(test_assoc, ordering, triples):
    if ordering is None:
        test_triples = test_assoc.triples()
        test_triples.sort()
        list_triples = list(triples)
        list_triples.sort()
        assert test_triples == list_triples
    else:
        assert test_assoc.triples(ordering=ordering) == triples


@pytest.mark.parametrize(
    "test_assoc,exp_dict",
    [
        (D4M.assoc.Assoc("a,b,", "A,B,", [2, 3]), {"a": {"A": 2}, "b": {"B": 3}}),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,"),
            {"a": {"A": "aA"}, "b": {"B": "bB"}},
        ),
        (
            D4M.assoc.Assoc("a,b,a,b,", "A,B,B,A,", "aA,bB,aB,bA,"),
            {"a": {"A": "aA", "B": "aB"}, "b": {"A": "bA", "B": "bB"}},
        ),
        (D4M.assoc.Assoc([], [], []), dict()),
        (
            D4M.assoc.Assoc([1, 3, 5, 3], [2, 4, 6, 2], [2, 12, 30, 6]),
            {1: {2: 2}, 3: {2: 6, 4: 12}, 5: {6: 30}},
        ),
    ],
)
def test_to_dict(test_assoc, exp_dict):
    assert test_assoc.to_dict() == exp_dict


@pytest.mark.parametrize(
    "test_assoc,exp_dict",
    [
        (D4M.assoc.Assoc("a,b,", "A,B,", [2, 3]), {("a", "A"): 2, ("b", "B"): 3}),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,"),
            {("a", "A"): "aA", ("b", "B"): "bB"},
        ),
        (
            D4M.assoc.Assoc("a,b,a,b,", "A,B,B,A,", "aA,bB,aB,bA,"),
            {("a", "A"): "aA", ("a", "B"): "aB", ("b", "A"): "bA", ("b", "B"): "bB"},
        ),
        (D4M.assoc.Assoc([], [], []), dict()),
        (
            D4M.assoc.Assoc([1, 3, 5, 3], [2, 4, 6, 2], [2, 12, 30, 6]),
            {(1, 2): 2, (3, 4): 12, (5, 6): 30, (3, 2): 6},
        ),
    ],
)
def test_to_dict2(test_assoc, exp_dict):
    assert test_assoc.to_dict2() == exp_dict


@pytest.mark.parametrize(
    "test_assoc",
    [
        (D4M.assoc.Assoc("a,b,", "A,B,", 1)),
        (D4M.assoc.Assoc("a,b,b,", "A,B,A,", 1)),
        (D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,")),
        (D4M.assoc.Assoc(["a", "b"], "A,B,", [1, 1])),
        (D4M.assoc.Assoc([1, 2], "A,B,", [1, 1])),
        (D4M.assoc.Assoc([], "A,B,", 1)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3])),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, "add")),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, D4M.util.first)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, "first")),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, D4M.util.last)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, "last")),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, min)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, max)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [3, 2, 3], None, D4M.util.times)),
        (
            D4M.assoc.Assoc(
                "a,b,a,", "A,B,B,", 1.0, sp.coo_matrix(np.array([[1.0, 3.0], [0, 2.0]]))
            )
        ),
        (
            D4M.assoc.Assoc(
                "a,b,a,",
                "A,B,B,",
                ["aA", "bA", "aB"],
                sp.coo_matrix(np.array([[1, 3], [0, 2]])),
            )
        ),
        (
            D4M.assoc.Assoc(
                "a,b,",
                "A,B,",
                ["aA", "bA", "aB"],
                sp.coo_matrix(np.array([[1, 3], [0, 2]])),
            )
        ),
        (
            D4M.assoc.Assoc(
                "a,b,",
                "A,B,",
                "aA,aB,bA,bB,",
                sp.coo_matrix(np.array([[1, 3], [4, 2]])),
                "unique",
            )
        ),
        (
            D4M.assoc.Assoc(
                np.array(["a", "b"]),
                np.array(["A", "B"]),
                np.array(["aA", "aB", "bA", "bB"]),
                sp.coo_matrix(np.array([[1, 3], [4, 2]])),
                "unique",
            )
        ),
        (
            D4M.assoc.Assoc(
                np.array(["a", "b"]),
                np.array(["B", "A"]),
                np.array(["aA", "aB", "bA", "bB"]),
                sp.coo_matrix(np.array([[1, 3], [4, 2]])),
                "unique",
            )
        ),
        (
            D4M.assoc.Assoc(
                "a,b,",
                "B,A,",
                "aA,aB,bA,bB,",
                sp.coo_matrix(np.array([[1, 3], [4, 2]])),
                "unique",
            )
        ),
    ],
)
def test_get_row(test_assoc):
    assert np.array_equal(test_assoc.get_row(), test_assoc.row)


@pytest.mark.parametrize(
    "test_assoc",
    [
        (D4M.assoc.Assoc("a,b,", "A,B,", 1)),
        (D4M.assoc.Assoc("a,b,b,", "A,B,A,", 1)),
        (D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,")),
        (D4M.assoc.Assoc(["a", "b"], "A,B,", [1, 1])),
        (D4M.assoc.Assoc([1, 2], "A,B,", [1, 1])),
        (D4M.assoc.Assoc([], "A,B,", 1)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3])),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, "add")),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, D4M.util.first)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, "first")),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, D4M.util.last)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, "last")),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, min)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, max)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [3, 2, 3], None, D4M.util.times)),
        (
            D4M.assoc.Assoc(
                "a,b,a,", "A,B,B,", 1.0, sp.coo_matrix(np.array([[1.0, 3.0], [0, 2.0]]))
            )
        ),
        (
            D4M.assoc.Assoc(
                "a,b,a,",
                "A,B,B,",
                ["aA", "bA", "aB"],
                sp.coo_matrix(np.array([[1, 3], [0, 2]])),
            )
        ),
        (
            D4M.assoc.Assoc(
                "a,b,",
                "A,B,",
                ["aA", "bA", "aB"],
                sp.coo_matrix(np.array([[1, 3], [0, 2]])),
            )
        ),
        (
            D4M.assoc.Assoc(
                "a,b,",
                "A,B,",
                "aA,aB,bA,bB,",
                sp.coo_matrix(np.array([[1, 3], [4, 2]])),
                "unique",
            )
        ),
        (
            D4M.assoc.Assoc(
                np.array(["a", "b"]),
                np.array(["A", "B"]),
                np.array(["aA", "aB", "bA", "bB"]),
                sp.coo_matrix(np.array([[1, 3], [4, 2]])),
                "unique",
            )
        ),
        (
            D4M.assoc.Assoc(
                np.array(["a", "b"]),
                np.array(["B", "A"]),
                np.array(["aA", "aB", "bA", "bB"]),
                sp.coo_matrix(np.array([[1, 3], [4, 2]])),
                "unique",
            )
        ),
        (
            D4M.assoc.Assoc(
                "a,b,",
                "B,A,",
                "aA,aB,bA,bB,",
                sp.coo_matrix(np.array([[1, 3], [4, 2]])),
                "unique",
            )
        ),
    ],
)
def test_get_col(test_assoc):
    assert np.array_equal(test_assoc.get_col(), test_assoc.col)


@pytest.mark.parametrize(
    "test_row,test_col,test_val,adj,aggregate,val",
    [
        ("a,b,", "A,B,", 1, None, None, np.array([1.0])),
        ("a,b,b,", "A,B,A,", 1, None, None, np.array([1.0])),
        ("a,b,", "A,B,", "aA,bB,", None, None, np.array(["aA", "bB"])),
        (["a", "b"], "A,B,", [1, 1], None, None, np.array([1])),
        ([1, 2], "A,B,", [1, 1], None, None, np.array([1])),
        ([], "A,B,", 1, None, None, np.empty(0)),
        ("a,b,a,", "A,B,A,", [1, 2, 3], None, D4M.util.add, np.array([2.0, 4.0])),
        ("a,b,a,", "A,B,A,", [1, 2, 3], None, D4M.util.first, np.array([1.0, 2.0])),
        ("a,b,a,", "A,B,A,", [1, 2, 3], None, D4M.util.last, np.array([2.0, 3.0])),
        ("a,b,a,", "A,B,A,", [1, 2, 3], None, min, np.array([1.0, 2.0])),
        ("a,b,a,", "A,B,A,", [1, 2, 3], None, max, np.array([2.0, 3.0])),
        ("a,b,a,", "A,B,A,", [3, 2, 3], None, D4M.util.times, np.array([2.0, 9.0])),
        (
            "a,b,a,",
            "A,B,B,",
            1.0,
            sp.coo_matrix(np.array([[1.0, 3.0], [0, 2.0]])),
            None,
            np.array([1.0, 2.0, 3.0]),
        ),
        (
            "a,b,a,",
            "A,B,B,",
            ["aA", "bA", "aB"],
            sp.coo_matrix(np.array([[1, 3], [0, 2]])),
            None,
            np.array(["aA", "aB", "bA"]),
        ),
        (
            "a,b,",
            "A,B,",
            ["aA", "bA", "aB"],
            sp.coo_matrix(np.array([[1, 3], [0, 2]])),
            None,
            np.array(["aA", "aB", "bA"]),
        ),
    ],
)
def test_get_val(test_row, test_col, test_val, adj, aggregate, val):
    adj, aggregate = _replace_default_args(D4M.assoc.Assoc, adj=adj, aggregate=aggregate)
    assoc_ = D4M.assoc.Assoc(test_row, test_col, test_val, adj=adj, aggregate=aggregate)
    assert np.array_equal(val, assoc_.get_val())


@pytest.mark.parametrize(
    "test_assoc",
    [
        (D4M.assoc.Assoc("a,b,", "A,B,", 1)),
        (D4M.assoc.Assoc("a,b,b,", "A,B,A,", 1)),
        (D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,")),
        (D4M.assoc.Assoc(["a", "b"], "A,B,", [1, 1])),
        (D4M.assoc.Assoc([1, 2], "A,B,", [1, 1])),
        (D4M.assoc.Assoc([], "A,B,", 1)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3])),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, "add")),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, D4M.util.first)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, "first")),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, D4M.util.last)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, "last")),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, min)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [1, 2, 3], None, max)),
        (D4M.assoc.Assoc("a,b,a,", "A,B,A,", [3, 2, 3], None, D4M.util.times)),
        (
            D4M.assoc.Assoc(
                "a,b,a,", "A,B,B,", 1.0, sp.coo_matrix(np.array([[1.0, 3.0], [0, 2.0]]))
            )
        ),
        (
            D4M.assoc.Assoc(
                "a,b,a,",
                "A,B,B,",
                ["aA", "bA", "aB"],
                sp.coo_matrix(np.array([[1, 3], [0, 2]])),
            )
        ),
        (
            D4M.assoc.Assoc(
                "a,b,",
                "A,B,",
                ["aA", "bA", "aB"],
                sp.coo_matrix(np.array([[1, 3], [0, 2]])),
            )
        ),
        (
            D4M.assoc.Assoc(
                "a,b,",
                "A,B,",
                "aA,aB,bA,bB,",
                sp.coo_matrix(np.array([[1, 3], [4, 2]])),
                "unique",
            )
        ),
        (
            D4M.assoc.Assoc(
                np.array(["a", "b"]),
                np.array(["A", "B"]),
                np.array(["aA", "aB", "bA", "bB"]),
                sp.coo_matrix(np.array([[1, 3], [4, 2]])),
                "unique",
            )
        ),
        (
            D4M.assoc.Assoc(
                np.array(["a", "b"]),
                np.array(["B", "A"]),
                np.array(["aA", "aB", "bA", "bB"]),
                sp.coo_matrix(np.array([[1, 3], [4, 2]])),
                "unique",
            )
        ),
        (
            D4M.assoc.Assoc(
                "a,b,",
                "B,A,",
                "aA,aB,bA,bB,",
                sp.coo_matrix(np.array([[1, 3], [4, 2]])),
                "unique",
            )
        ),
    ],
)
def test_get_adj(test_assoc):
    assert D4M.assoc.sparse_equal(test_assoc.get_adj(), test_assoc.adj)


@pytest.mark.parametrize(
    "test_assoc,row_key,col_key,value",
    [
        (D4M.assoc.Assoc("a,b,", "A,B,", [2, 3]), "b", "B", 3),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,color,color,", "amber,auburn,white,"
            ),
            "amber",
            "color",
            "amber",
        ),
        (D4M.assoc.Assoc("a,b,a,b,", "A,B,B,A,", "aA,bB,aB,bA,"), "a", "B", "aB"),
        (D4M.assoc.Assoc("a,b,", "A,B,", [2, 3]), "c", "B", 0),
    ],
)
def test_get_value(test_assoc, row_key, col_key, value):
    assert test_assoc.get_value(row_key, col_key) == value


@pytest.mark.parametrize(
    "test_assoc,subrow,subcol,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", [2, 3]),
            "a,",
            "A,B,",
            D4M.assoc.Assoc("a,", "A,", [2]),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,"),
            "a,",
            "A,B,",
            D4M.assoc.Assoc("a,", "A,", "aA,"),
        ),
        (
            D4M.assoc.Assoc("a,b,a,b,", "A,B,B,A,", "aA,bB,aB,bA,"),
            "a,b,",
            "A,",
            D4M.assoc.Assoc("a,b,", "A,A,", "aA,bA,"),
        ),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,color,color,", "amber,auburn,white,"
            ),
            D4M.util.startswith("a,"),
            ":",
            D4M.assoc.Assoc("amber,ash,", "color,color,", "amber,auburn,"),
        ),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,height,leaf_color,", ["amber", 1.7, "green"]
            ),
            ":",
            D4M.util.contains("color,"),
            D4M.assoc.Assoc("amber,birch,", "color,leaf_color,", ["amber", "green"]),
        ),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,color,color,", "amber,auburn,white,"
            ),
            [0, 1],
            ":",
            D4M.assoc.Assoc("amber,ash,", "color,color,", "amber,auburn,"),
        ),
        (
            D4M.assoc.Assoc("amber,ash,", "color,color,", "amber,auburn,"),
            D4M.util.startswith("c,"),
            ":",
            D4M.assoc.Assoc([], [], []),
        ),
        (
            D4M.assoc.Assoc("amber,ash,", "color,color,", "amber,auburn,"),
            0,
            ":",
            D4M.assoc.Assoc("amber,", "color,", "amber,"),
        ),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,color,color,", "amber,auburn,white,"
            ),
            np.array([0, 1]),
            ":",
            D4M.assoc.Assoc("amber,ash,", "color,color,", "amber,auburn,"),
        ),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,size,beauty,", "amber,big,ugly,"
            ),
            ":",
            np.array([0, 2]),
            D4M.assoc.Assoc("ash,birch,", "size,beauty,", "big,ugly,"),
        ),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,color,color,", "amber,auburn,white,"
            ),
            slice(0, 2, 1),
            slice(None, None, None),
            D4M.assoc.Assoc("amber,ash,", "color,color,", "amber,auburn,"),
        ),
    ],
)
def test_getitem(test_assoc, subrow, subcol, exp_assoc):
    test_assoc.printfull()
    print(subrow)
    print(subcol)
    sub_assoc = test_assoc[subrow, subcol]
    assert D4M.assoc.assoc_equal(sub_assoc, exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc,rownum,colnum",
    [
        (D4M.assoc.Assoc("a,b,", "A,B,", [2, 3]), 2, 2),
        (D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,"), 2, 2),
        (D4M.assoc.Assoc("a,b,a,b,", "A,B,B,A,", "aA,bB,aB,bA,"), 2, 2),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,color,color,", "amber,auburn,white,"
            ),
            3,
            1,
        ),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,height,leaf_color,", ["amber", 1.7, "green"]
            ),
            3,
            3,
        ),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,color,color,", "amber,auburn,white,"
            ),
            3,
            1,
        ),
        (D4M.assoc.Assoc("amber,ash,", "color,color,", "amber,auburn,"), 2, 1),
        (D4M.assoc.Assoc([], [], []), 0, 0),
    ],
)
def test_size(test_assoc, rownum, colnum):
    test_rownum, test_colnum = test_assoc.size()
    assert test_rownum == rownum
    assert test_colnum == colnum


@pytest.mark.parametrize(
    "test_assoc,exp_nnz",
    [
        (D4M.assoc.Assoc("a,b,", "A,B,", [2, 0]), 1),
        (D4M.assoc.Assoc("a,b,", "A,B,", 2), 2),
        (D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,"), 2),
        (D4M.assoc.Assoc("a,b,a,b,", "A,B,B,A,", "aA,bB,aB,bA,"), 4),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,color,color,", "amber,auburn,white,"
            ),
            3,
        ),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,height,leaf_color,", ["amber", 1.7, "green"]
            ),
            3,
        ),
        (D4M.assoc.Assoc("amber,ash,birch,", "color,color,color,", "amber,,white,"), 2),
        (D4M.assoc.Assoc("amber,ash,", "color,color,", "amber,auburn,"), 2),
        (D4M.assoc.Assoc("a,", "A,", [""]), 0),
        (D4M.assoc.Assoc([], [], []), 0),
    ],
)
def test_nnz(test_assoc, exp_nnz):
    assert test_assoc.nnz() == exp_nnz


@pytest.mark.parametrize(
    "test_assoc",
    [
        (D4M.assoc.Assoc("a,b,", "A,B,", [2, 3])),
        (D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,")),
        (D4M.assoc.Assoc("a,b,a,b,", "A,B,B,A,", "aA,bB,aB,bA,")),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,color,color,", "amber,auburn,white,"
            )
        ),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,height,leaf_color,", ["amber", 1.7, "green"]
            )
        ),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,color,color,", "amber,auburn,white,"
            )
        ),
        (D4M.assoc.Assoc("amber,ash,", "color,color,", "amber,auburn,")),
        (D4M.assoc.Assoc("amber,ash,", "color,color,", "amber,auburn,")),
        (D4M.assoc.Assoc([], [], [])),
    ],
)
def test_copy(test_assoc):
    exp_assoc = test_assoc.copy()

    assert D4M.assoc.assoc_equal(test_assoc, exp_assoc, return_info=True)

    assert not (test_assoc.row is exp_assoc.row)
    assert not (test_assoc.col is exp_assoc.col)
    if isinstance(test_assoc.val, float) or isinstance(exp_assoc.val, float):
        assert test_assoc.val == exp_assoc.val
    else:
        assert not (test_assoc.val is exp_assoc.val)
    assert not (test_assoc.adj is exp_assoc.adj)


@pytest.mark.parametrize(
    "test_assoc",
    [
        (D4M.assoc.Assoc("a,b,", "A,B,", [2, 3])),
        (D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,")),
        (D4M.assoc.Assoc("a,b,a,b,", "A,B,B,A,", "aA,bB,aB,bA,")),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,color,color,", "amber,auburn,white,"
            )
        ),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,height,leaf_color,", ["amber", 1.7, "green"]
            )
        ),
        (
            D4M.assoc.Assoc(
                "amber,ash,birch,", "color,color,color,", "amber,auburn,white,"
            )
        ),
        (D4M.assoc.Assoc("amber,ash,", "color,color,", "amber,auburn,")),
        (D4M.assoc.Assoc("amber,ash,", "color,color,", "amber,auburn,")),
        (D4M.assoc.Assoc([], [], [])),
    ],
)
def test_deepcopy(test_assoc):
    exp_assoc = test_assoc.deepcopy()

    assert D4M.assoc.assoc_equal(test_assoc, exp_assoc, return_info=True)

    assert not (test_assoc.row is exp_assoc.row)
    assert not (test_assoc.col is exp_assoc.col)
    if not isinstance(test_assoc.val, float):
        assert not (test_assoc.val is exp_assoc.val)
    assert not (test_assoc.adj.data is exp_assoc.adj.data)
    assert not (test_assoc.adj.row is exp_assoc.adj.row)
    assert not (test_assoc.adj.col is exp_assoc.adj.col)


@pytest.mark.parametrize(
    "test_assoc,exp_assoc",
    [
        (D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1), D4M.assoc.Assoc([], [], [])),
        (D4M.assoc.Assoc("a,b,b,", "a,a,b,", 1), D4M.assoc.Assoc("a,b,", "a,b,", 1)),
        (
            D4M.assoc.Assoc([1, 2, 2, 5], [1, 3, 2, 5], [2.4, 1, 9.7, -1]),
            D4M.assoc.Assoc([1, 2, 5], [1, 2, 5], [2.4, 9.7, -1]),
        ),
        (
            D4M.assoc.Assoc([1, 2, 2, 5], [1, 3, 2, 5], "1-1,2-3,2-2,5-5,"),
            D4M.assoc.Assoc([1, 2, 5], [1, 2, 5], "1-1,2-2,5-5,"),
        ),
        (
            D4M.assoc.Assoc([1, 2, 3, 5], [1, 3, 2, 5], "1-1,2-3,3-2,5-5,"),
            D4M.assoc.Assoc([1, 5], [1, 5], "1-1,5-5,"),
        ),
        (D4M.assoc.Assoc([], [], []), D4M.assoc.Assoc([], [], [])),
    ],
)
def test_diag(test_assoc, exp_assoc):
    assert D4M.assoc.assoc_equal(test_assoc.diag(), exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1),
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "a,a,b,", 2),
            D4M.assoc.Assoc("a,b,b,", "a,a,b,", 1),
        ),
        (
            D4M.assoc.Assoc([1, 2, 2, 5], [1, 3, 2, 5], [2.4, 1, 9.7, -1]),
            D4M.assoc.Assoc([1, 2, 2, 5], [1, 3, 2, 5], 1),
        ),
        (
            D4M.assoc.Assoc([1, 2, 2, 5], [1, 3, 2, 5], "1-1,2-3,2-2,5-5,"),
            D4M.assoc.Assoc([1, 2, 2, 5], [1, 3, 2, 5], 1),
        ),
        (
            D4M.assoc.Assoc([1, 2, 3, 5], [1, 3, 2, 5], "1-1,2-3,3-2,5-5,"),
            D4M.assoc.Assoc([1, 2, 3, 5], [1, 3, 2, 5], 1),
        ),
        (D4M.assoc.Assoc([], [], []), D4M.assoc.Assoc([], [], [])),
    ],
)
def test_logical(test_assoc, exp_assoc):
    assert D4M.assoc.assoc_equal(test_assoc.logical(), exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1),
            D4M.assoc.Assoc("A,A,B,", "a,b,b,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "a,a,b,", 2),
            D4M.assoc.Assoc("a,a,b,", "a,b,b,", 2),
        ),
        (
            D4M.assoc.Assoc([1, 2, 2, 5], [1, 3, 2, 5], [2.4, 1, 9.7, -1]),
            D4M.assoc.Assoc([1, 3, 2, 5], [1, 2, 2, 5], [2.4, 1, 9.7, -1]),
        ),
        (
            D4M.assoc.Assoc([1, 2, 2, 5], [1, 3, 2, 5], "1-1,2-3,2-2,5-5,"),
            D4M.assoc.Assoc([1, 3, 2, 5], [1, 2, 2, 5], "1-1,2-3,2-2,5-5,"),
        ),
        (
            D4M.assoc.Assoc([1, 2, 3, 5], [1, 3, 2, 5], "1-1,2-3,3-2,5-5,"),
            D4M.assoc.Assoc([1, 3, 2, 5], [1, 2, 3, 5], "1-1,2-3,3-2,5-5,"),
        ),
        (D4M.assoc.Assoc([], [], []), D4M.assoc.Assoc([], [], [])),
    ],
)
def test_transpose(test_assoc, exp_assoc):
    assert D4M.assoc.assoc_equal(test_assoc.transpose(), exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc,exp_assoc",
    [
        (D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1), D4M.assoc.Assoc("a,b,", 0, 1)),
        (D4M.assoc.Assoc([], [], []), D4M.assoc.Assoc([], [], [])),
        (D4M.assoc.Assoc("a,b,c,d,", "A,", 2), D4M.assoc.Assoc("a,b,c,d,", 0, 1)),
    ],
)
def test_nocol(test_assoc, exp_assoc):
    assert D4M.assoc.assoc_equal(test_assoc.nocol(), exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc,exp_assoc",
    [
        (D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1), D4M.assoc.Assoc(0, "A,B,", 1)),
        (D4M.assoc.Assoc([], [], []), D4M.assoc.Assoc([], [], [])),
        (D4M.assoc.Assoc("a,", "A,B,C,D,", 2), D4M.assoc.Assoc(0, "A,B,C,D,", 1)),
    ],
)
def test_norow(test_assoc, exp_assoc):
    assert D4M.assoc.assoc_equal(test_assoc.norow(), exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc,axis,exp_assoc",
    [
        (D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1), None, 3),
        (D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1), 1, D4M.assoc.Assoc("a,b,", 0, [1, 2])),
        (D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1), 0, D4M.assoc.Assoc(0, "A,B,", [2, 1])),
        (D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA,bA,bB,"), None, 3),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA,bA,bB,"),
            1,
            D4M.assoc.Assoc("a,b,", 0, [1, 2]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA,bA,bB,"),
            0,
            D4M.assoc.Assoc(0, "A,B,", [2, 1]),
        ),
        (D4M.assoc.Assoc("a,b,b,c,c,", "A,A,B,A,B,", [7, -1, 4.5, 3, -3]), None, 10.5),
        (
            D4M.assoc.Assoc("a,b,b,c,c,", "A,A,B,A,B,", [7, -1, 4.5, 3, -3]),
            1,
            D4M.assoc.Assoc("a,b,", 0, [7, 3.5]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,c,c,", "A,A,B,A,B,", [7, -1, 4.5, 3, -3]),
            0,
            D4M.assoc.Assoc(0, "A,B,", [9, 1.5]),
        ),
    ],
)
def test_sum(test_assoc, axis, exp_assoc):
    if axis is None:
        assert isinstance(test_assoc.sum(), Number)
        assert test_assoc.sum() == exp_assoc
    else:
        assert isinstance(exp_assoc, D4M.assoc.Assoc)
        assert D4M.assoc.assoc_equal(test_assoc.sum(axis), exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,binary_op,right_zero,left_zero,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.util.add,
            None,
            None,
            D4M.assoc.Assoc("a,b,", "A,B,", 2),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,C,", 1),
            D4M.util.add,
            None,
            None,
            D4M.assoc.Assoc("a,b,b,", "A,B,C,", [2, 1, 1]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA,bA,bB,"),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA,aB,bB,"),
            D4M.util.add,
            None,
            None,
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aAaA,aB,bA,bBbB,"),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 2),
            D4M.assoc.Assoc("a,b,", "A,B,", 3),
            D4M.util.times,
            None,
            None,
            D4M.assoc.Assoc("a,b,", "A,B,", 6),
        ),
        (
            D4M.assoc.Assoc("a,b,c,", "A,B,C,", 2),
            D4M.assoc.Assoc("a,b,d,", "A,B,D,", 3),
            D4M.util.times,
            True,
            True,
            D4M.assoc.Assoc("a,b,", "A,B,", 6),
        ),
    ],
)
def test_combine(
    test_assoc_1, test_assoc_2, binary_op, right_zero, left_zero, exp_assoc
):
    combined_assoc = test_assoc_1.combine(
        test_assoc_2, binary_op, right_zero=right_zero, left_zero=left_zero
    )
    assert D4M.assoc.assoc_equal(combined_assoc, exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,semi_add,semi_mult,exp_assoc",
    [
        (
            D4M.assoc.Assoc([1, 1, 3], [2, 4, 6], [2, 4, 18]),
            D4M.assoc.Assoc([2, 3, 4, 4], ["a", "a", "b", "c"], 1),
            D4M.util.add,
            D4M.util.times,
            D4M.assoc.Assoc([1, 1, 1], ["a", "b", "c"], [2, 4, 4]),
        ),
        (
            D4M.assoc.Assoc("a,d,b,b,c,d,a,", [1, 3, 7, 1, 5, 2, 1], 1),
            D4M.assoc.Assoc([1, 2, 3, 5, 7], "A,B,C,D,E,", 1),
            D4M.util.add,
            D4M.util.times,
            D4M.assoc.Assoc("a,d,b,b,c,d,a,", "A,C,E,A,D,B,A,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc([], [], []),
            D4M.util.add,
            D4M.util.times,
            D4M.assoc.Assoc([], [], []),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.util.add,
            D4M.util.times,
            D4M.assoc.Assoc([], [], []),
        ),
        (
            D4M.assoc.Assoc([1, 1, 3], [2, 4, 6], [2, 4, 18]),
            D4M.assoc.Assoc([2, 3, 4, 4], ["a", "a", "b", "c"], 1),
            D4M.util.add,
            D4M.util.times,
            D4M.assoc.Assoc([1, 1, 1], ["a", "b", "c"], [2, 4, 4]),
        ),
        (
            D4M.assoc.Assoc([1, 1, 3], [2, 4, 6], [2, 4, 18]),
            D4M.assoc.Assoc([2, 3, 4, 4], ["a", "a", "b", "c"], [-1, 4, 2, 7]),
            D4M.util.add,
            D4M.util.times,
            D4M.assoc.Assoc([1, 1, 1], ["a", "b", "c"], [-2, 8, 28]),
        ),
    ],
)
def test_semiring_prod(test_assoc_1, test_assoc_2, semi_add, semi_mult, exp_assoc):
    prod = test_assoc_1.semiring_prod(test_assoc_2, semi_add, semi_mult)
    assert D4M.assoc.assoc_equal(prod, exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 2),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,C,", 1),
            D4M.assoc.Assoc("a,b,b,", "A,B,C,", [2, 1, 1]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA,bA,bB,"),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA,aB,bB,"),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aAaA,aB,bA,bBbB,"),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA,bA,bB,"),
            D4M.assoc.Assoc([], [], []),
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA,bA,bB,"),
        ),
    ],
)
def test_add(test_assoc_1, test_assoc_2, exp_assoc):
    assert D4M.assoc.assoc_equal(
        test_assoc_1 + test_assoc_2, exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc([], [], []),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,C,", 1),
            D4M.assoc.Assoc("b,b,", "B,C,", [1, -1]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA,bA,bB,"),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA,aB,bB,"),
            D4M.assoc.Assoc("b,", "A,", "bA,"),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA,bA,bB,"),
            D4M.assoc.Assoc([], [], []),
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA,bA,bB,"),
        ),
    ],
)
def test_sub(test_assoc_1, test_assoc_2, exp_assoc):
    assert D4M.assoc.assoc_equal(
        test_assoc_1 - test_assoc_2, exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc([], [], []),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("A,B,", "a,b,", 1),
            D4M.assoc.Assoc("a,b,", "a,b,", 1),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            D4M.assoc.Assoc("A,A,B,", "a,b,b,", 1),
            D4M.assoc.Assoc("a,a,b,", "a,b,b,", [1, 2, 1]),
        ),
        (
            D4M.assoc.Assoc("a,b,c,", "A,B,C,", "aA,bB,cC,"),
            D4M.assoc.Assoc("A,B,D,", "a,b,d,", "Aa,Bb,Dd,"),
            D4M.assoc.Assoc("a,b,", "a,b,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc([], [], []),
            D4M.assoc.Assoc([], [], []),
        ),
        (
            D4M.assoc.Assoc([1, 1, 3], [2, 4, 6], [2, 4, 18]),
            D4M.assoc.Assoc([2, 3, 4, 4], ["a", "a", "b", "c"], 1),
            D4M.assoc.Assoc([1, 1, 1], ["a", "b", "c"], [2, 4, 4]),
        ),
        (
            D4M.assoc.Assoc("a,d,b,b,c,d,a,", [1, 3, 7, 1, 5, 2, 1], 1),
            D4M.assoc.Assoc([1, 2, 3, 5, 7], "A,B,C,D,E,", 1),
            D4M.assoc.Assoc("a,d,b,b,c,d,a,", "A,C,E,A,D,B,A,", 1),
        ),
    ],
)
def test_matmul(test_assoc_1, test_assoc_2, exp_assoc):
    assert D4M.assoc.assoc_equal(
        test_assoc_1 @ test_assoc_2, exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 2),
            D4M.assoc.Assoc("a,b,", "A,B,", 3),
            D4M.assoc.Assoc("a,b,", "A,B,", 6),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1),
            D4M.assoc.Assoc([], [], []),
            D4M.assoc.Assoc([], [], []),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("b,a,", "A,B,", 1),
            D4M.assoc.Assoc([], [], []),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,", "A,", "aA,"),
            D4M.assoc.Assoc("a,", "A,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,"),
            D4M.assoc.Assoc("a,", "A,", 1),
            D4M.assoc.Assoc("a,", "A,", "aA,"),
        ),
        (
            D4M.assoc.Assoc("1,2,2,", "0,0,1,", [2, 3, 4]),
            D4M.assoc.Assoc("1,1,2,", "0,1,1,", [2, 3, 4]),
            D4M.assoc.Assoc("1,2,", "0,1,", [4, 16]),
        ),
        (
            D4M.assoc.Assoc("a,b,c,", "A,B,C,", 1),
            D4M.assoc.Assoc("a,b,d,", "A,B,D,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,c,", "A,B,C,", 2),
            D4M.assoc.Assoc("a,c,", "A,C,", "aA,cC,"),
            D4M.assoc.Assoc("a,c,", "A,C,", 2),
        ),
        (
            D4M.assoc.Assoc("a,b,c,", "A,B,C,", "Aa,Bb,Cc,"),
            D4M.assoc.Assoc("a,c,", "A,C,", "aA,cC,"),
            D4M.assoc.Assoc("a,c,", "A,C,", "Aa,Cc,"),
        ),
        (
            D4M.assoc.Assoc("a,b,c,", "A,B,C,", "Aa,Bb,Cc,"),
            D4M.assoc.Assoc("a,c,", "A,C,", 2),
            D4M.assoc.Assoc("a,c,", "A,C,", "Aa,Cc,"),
        ),
    ],
)
def test_elementwise_multiply(test_assoc_1, test_assoc_2, exp_assoc):
    assert D4M.assoc.assoc_equal(
        test_assoc_1 * test_assoc_2, exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 2),
            D4M.assoc.Assoc("a,b,", "A,B,", 3),
            D4M.assoc.Assoc("a,b,", "A,B,", 2 / 3),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1),
            D4M.assoc.Assoc([], [], []),
            D4M.assoc.Assoc([], [], []),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("b,a,", "A,B,", 1),
            D4M.assoc.Assoc([], [], []),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,", "A,", "aA,"),
            D4M.assoc.Assoc("a,", "A,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,"),
            D4M.assoc.Assoc("a,", "A,", 1),
            D4M.assoc.Assoc("a,", "A,", 1),
        ),
        (
            D4M.assoc.Assoc("1,2,2,", "0,0,1,", [2, 3, 4]),
            D4M.assoc.Assoc("1,1,2,", "0,1,1,", [2, 3, 4]),
            D4M.assoc.Assoc("1,2,", "0,1,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,c,", "A,B,C,", 1),
            D4M.assoc.Assoc("a,b,d,", "A,B,D,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,c,", "A,B,C,", 2),
            D4M.assoc.Assoc("a,c,", "A,C,", "aA,cC,"),
            D4M.assoc.Assoc("a,c,", "A,C,", 2),
        ),
        (
            D4M.assoc.Assoc("a,b,c,", "A,B,C,", "Aa,Bb,Cc,"),
            D4M.assoc.Assoc("a,c,", "A,C,", "aA,cC,"),
            D4M.assoc.Assoc("a,c,", "A,C,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,c,", "A,B,C,", "Aa,Bb,Cc,"),
            D4M.assoc.Assoc("a,c,", "A,C,", 2),
            D4M.assoc.Assoc("a,c,", "A,C,", 1 / 2),
        ),
    ],
)
def test_elementwise_divide(test_assoc_1, test_assoc_2, exp_assoc):
    assert D4M.assoc.assoc_equal(
        test_assoc_1.divide(test_assoc_2), exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "test_assoc,scalar,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]),
            3,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [3, 6, 9]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]),
            2.3,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [2.3, 4.6, 6.9])
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]),
            -3,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [-3, -6, -9]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]),
            -2.3,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [-2.3, -4.6, -6.9])
        ),
        (
            3,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [3, 6, 9]),
        ),
        (
            2.3,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [2.3, 4.6, 6.9])
        ),
        (
            -3,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [-3, -6, -9]),
        ),
        (
            -2.3,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [-2.3, -4.6, -6.9])
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA,aB,bB,"),
            3,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 3),
        ),
        (D4M.assoc.Assoc([], [], []), 3, D4M.assoc.Assoc([], [], [])),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]),
            0,
            D4M.assoc.Assoc([], [], []),
        ),
    ],
)
def test_mul_scalar(test_assoc, scalar, exp_assoc):
    assert D4M.assoc.assoc_equal(test_assoc @ scalar, exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc,scalar,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]),
            3,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1 / 3, 2 / 3, 1]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]),
            3.1,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1 / 3.1, 2 / 3.1, 3 / 3.1]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]),
            -3.1,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [-1 / 3.1, -2 / 3.1, -3 / 3.1]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA,aB,bB,"),
            3,
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1 / 3),
        ),
        (D4M.assoc.Assoc([], [], []), 3, D4M.assoc.Assoc([], [], [])),
    ],
)
def test_divide_scalar(test_assoc, scalar, exp_assoc):
    assert D4M.assoc.assoc_equal(test_assoc.divide(scalar), exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc", [(D4M.assoc.Assoc("a,a,b,", "A,B,B,", [1, 2, 3]))]
)
def test_divide_by_zero(test_assoc):
    with pytest.raises(Exception) as e_info:
        test_assoc.divide(0)
    assert str(e_info.value) == "division by zero"


def num_lex_sort(num_1: Number, num_2: Number) -> bool:
    num_str_1 = str(num_1)
    num_str_2 = str(num_2)
    return num_str_1 < num_str_2


def shortlex_sort(str_1: str, str_2: str) -> bool:
    if len(str_1) < len(str_2):
        return True
    elif len(str_1) > len(str_2):
        return False
    else:
        return str_1 < str_2


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,sort_key,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 2),
            None,
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [-1, 1, 2]),
            D4M.assoc.Assoc([], [], []),
            None,
            D4M.assoc.Assoc("a,", "A,", [-1]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA1,bA1,bB2,"),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA2,aB2,bB1,"),
            None,
            D4M.assoc.Assoc("a,b,", "A,B,", "aA1,bB1,"),
        ),
        (
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aA1,,bA1,bB2,"),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aA2,aB2,,bB1,"),
            str.__lt__,
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aA1,,,bB1,"),
        ),
        (
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [10, 0, 2, -2]),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [2, -1, 0, -1]),
            num_lex_sort,
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [10, -1, 0, -1]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,c,d,", "A,A,B,C,D,", "aAA,bAA,bZ,cC,,"),
            D4M.assoc.Assoc("a,b,b,c,d,", "A,A,B,C,D,", "aZ,bZ,bBB,,dD,"),
            shortlex_sort,
            D4M.assoc.Assoc("a,b,b,c,d,", "A,A,B,C,D,", "aZ,bZ,bZ,,,"),
        ),
    ],
)
def test_min(test_assoc_1, test_assoc_2, sort_key, exp_assoc):
    assert D4M.assoc.assoc_equal(
        test_assoc_1.min(test_assoc_2, sort_key=sort_key), exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,sort_key,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 2),
            None,
            D4M.assoc.Assoc("a,b,", "A,B,", 2),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [-1, 1, 2]),
            D4M.assoc.Assoc([], [], []),
            None,
            D4M.assoc.Assoc("b,b,", "A,B,", [1, 2]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA1,bA1,bB2,"),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA2,aB2,bB1,"),
            None,
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aA2,aB2,bA1,bB2,"),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA1,bA1,bB2,"),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA2,aB2,bB1,"),
            str.__lt__,
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aA2,aB2,bA1,bB2,"),
        ),
        (
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [10, 0, 2, -2]),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [2, -1, 0, -1]),
            num_lex_sort,
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [2, 0, 2, -2]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,c,d,", "A,A,B,C,D,", "aAA,bAA,bZ,cC,,"),
            D4M.assoc.Assoc("a,b,b,c,d,", "A,A,B,C,D,", "aZ,bZ,bBB,,dD,"),
            shortlex_sort,
            D4M.assoc.Assoc("a,b,b,c,d,", "A,A,B,C,D,", "aAA,bAA,bBB,cC,dD,"),
        ),
    ],
)
def test_max(test_assoc_1, test_assoc_2, sort_key, exp_assoc):
    assert D4M.assoc.assoc_equal(
        test_assoc_1.max(test_assoc_2, sort_key=sort_key), exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA,bA,bB,"),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA,aB,bB,"),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
        ),
        (
            D4M.assoc.Assoc("a,", "A,", 1),
            D4M.assoc.Assoc("b,", "B,", 1),
            D4M.assoc.Assoc([], [], []),
        ),
    ],
)
def test_and(test_assoc_1, test_assoc_2, exp_assoc):
    assert D4M.assoc.assoc_equal(test_assoc_1 & test_assoc_2, exp_assoc)


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA,bA,bB,"),
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", "aA,aB,bB,"),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", 1),
        ),
        (
            D4M.assoc.Assoc("a,", "A,", 1),
            D4M.assoc.Assoc("b,", "B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
        ),
    ],
)
def test_or(test_assoc_1, test_assoc_2, exp_assoc):
    assert D4M.assoc.assoc_equal(test_assoc_1 | test_assoc_2, exp_assoc)


@pytest.mark.parametrize(
    "test_assoc,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", 1),
            D4M.assoc.Assoc("A,B,B,A,", "A,A,B,B,", [2, 1, 1, 1]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", [2, 3, 1]),
            D4M.assoc.Assoc("A,A,B,B,", "A,B,A,B,", [4, 6, 6, 10]),
        ),
        (
            D4M.assoc.Assoc([2, 2, 4, 6], [1, 3, 3, 1], 1),
            D4M.assoc.Assoc([1, 1, 3, 3], [1, 3, 1, 3], [2, 1, 1, 2]),
        ),
    ],
)
def test_sqin(test_assoc, exp_assoc):
    assert D4M.assoc.assoc_equal(test_assoc.sqin(), exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,B,", 1),
            D4M.assoc.Assoc("a,b,b,a,", "a,a,b,b,", [2, 1, 1, 1]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [2, 3, 1]),
            D4M.assoc.Assoc("a,a,b,b,", "a,b,a,b,", [4, 6, 6, 10]),
        ),
        (
            D4M.assoc.Assoc([2, 2, 4, 6], [1, 3, 3, 1], 1),
            D4M.assoc.Assoc(
                [2, 2, 2, 4, 4, 6, 6], [2, 4, 6, 2, 4, 2, 6], [2, 1, 1, 1, 1, 1, 1]
            ),
        ),
    ],
)
def test_sqout(test_assoc, exp_assoc):
    assert D4M.assoc.assoc_equal(test_assoc.sqout(), exp_assoc, return_info=True)


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,delimiter,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,A,", 1),
            D4M.assoc.Assoc("A,A,B,", "a,b,b,", 1),
            None,
            D4M.assoc.Assoc("a,a,b,b,", "a,b,a,b,", "A;,A;B;,A;,A;,"),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,A,", "aA,aB,bA,"),
            D4M.assoc.Assoc("A,A,B,", "a,b,b,", 2),
            None,
            D4M.assoc.Assoc("a,a,b,b,", "a,b,a,b,", "A;,A;B;,A;,A;,"),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,A,", 1),
            D4M.assoc.Assoc("A,A,B,", "a,b,b,", 1),
            "|",
            D4M.assoc.Assoc("a,a,b,b,", "a,b,a,b,", "A|,A|B|,A|,A|,"),
        ),
        (
            D4M.assoc.Assoc("a,a,b,c,", "A,B,A,C,", 1),
            D4M.assoc.Assoc("A,A,B,D,", "a,b,b,d,", 1),
            "+",
            D4M.assoc.Assoc("a,a,b,b,", "a,b,a,b,", "A+,A+B+,A+,A+,"),
        ),
        (
            D4M.assoc.Assoc([1, 1, 3, 3], [2, 4, 2, 10], [1, 1, 2, 1]),
            D4M.assoc.Assoc([2, 2, 4, 10, 10], [1, 3, 1, 1, 3], [-10, 2, 10, 10, 1]),
            None,
            D4M.assoc.Assoc([1, 1, 3, 3], [1, 3, 1, 3], "2;4;,2;,2;10;,2;10;,"),
        ),
    ],
)
def test_catkeymul(test_assoc_1, test_assoc_2, delimiter, exp_assoc):
    delimiter = _replace_default_args(D4M.assoc.Assoc.catkeymul, delimiter=delimiter)
    assert D4M.assoc.assoc_equal(
        test_assoc_1.catkeymul(test_assoc_2, delimiter=delimiter),
        exp_assoc,
        return_info=True,
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,pair_delimiter,delimiter,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,A,", [1, 2, 3]),
            D4M.assoc.Assoc("A,A,B,", "a,b,b,", [3, 2, 1]),
            None,
            None,
            D4M.assoc.Assoc(
                "a,a,b,b,", "a,b,a,b,", ["1,3,;", "1,2,;2,1,;", "3,3,;", "3,2,;"]
            ),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,A,", "aA,aB,bA,"),
            D4M.assoc.Assoc("A,A,B,", "a,b,b,", [3, 2, 1]),
            None,
            None,
            D4M.assoc.Assoc(
                "a,a,b,b,", "a,b,a,b,", ["aA,3,;", "aA,2,;aB,1,;", "bA,3,;", "bA,2,;"]
            ),
        ),
        (
            D4M.assoc.Assoc("a,a,b,", "A,B,A,", "aA,aB,bA,"),
            D4M.assoc.Assoc("A,A,B,", "a,b,b,", "Aa,Ab,Bb,"),
            "|",
            None,
            D4M.assoc.Assoc(
                "a,a,b,b,",
                "a,b,a,b,",
                ["aA|Aa|;", "aA|Ab|;aB|Bb|;", "bA|Aa|;", "bA|Ab|;"],
            ),
        ),
        (
            D4M.assoc.Assoc("a,a,b,c,", "A,B,A,C,", "aA,aB,bA,cC,"),
            D4M.assoc.Assoc("A,A,B,D,", "a,b,b,d,", "Aa,Ab,Bb,Dd,"),
            "*",
            "+",
            D4M.assoc.Assoc(
                "a,a,b,b,",
                "a,b,a,b,",
                ["aA*Aa*+", "aA*Ab*+aB*Bb*+", "bA*Aa*+", "bA*Ab*+"],
            ),
        ),
        (
            D4M.assoc.Assoc([1, 1, 3, 3], [2, 4, 2, 10], [1, 1, 2, 1]),
            D4M.assoc.Assoc([2, 2, 4, 10, 10], [1, 3, 1, 1, 3], [-10.5, 2, 10, 10, 1]),
            None,
            "&",
            D4M.assoc.Assoc(
                [1, 1, 3, 3],
                [1, 3, 1, 3],
                ["1,-10.5,&1,10,&", "1,2,&", "2,-10.5,&1,10,&", "2,2,&1,1,&"],
            ),
        ),
    ],
)
def test_catvalmul(test_assoc_1, test_assoc_2, pair_delimiter, delimiter, exp_assoc):
    pair_delimiter, delimiter = _replace_default_args(
        D4M.assoc.Assoc.catvalmul, pair_delimiter=pair_delimiter, delimiter=delimiter
    )
    assert D4M.assoc.assoc_equal(
        test_assoc_1.catvalmul(
            test_assoc_2, pair_delimiter=pair_delimiter, delimiter=delimiter
        ),
        exp_assoc,
        return_info=True,
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,sort_key,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [10, 0, 2, -1, 1]),
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [2, -1, 0, -2, 1]),
            num_lex_sort,
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [1, 0, 0, 1, 0]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [2, -1, 0, -2, 1]),
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [10, 0, 2, -1, 1]),
            num_lex_sort,
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [0, 1, 1, 0, 0]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aAA,bAA,bZ,cC,,eE,"),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aZ,bZ,bBB,,dD,eE,"),
            shortlex_sort,
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", [0, 0, 1, 0, 1, 0]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aZ,bZ,bBB,,dD,eE,"),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aAA,bAA,bZ,cC,,eE,"),
            shortlex_sort,
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", [1, 1, 0, 1, 0, 0]),
        ),
    ],
)
def test_compare(test_assoc_1, test_assoc_2, sort_key, exp_assoc):
    assert D4M.assoc.assoc_equal(
        test_assoc_1.compare(test_assoc_2, sort_key), exp_assoc
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,sort_key,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [10, 0, 2, -1, 1]),
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [2, -1, 0, -2, 1]),
            num_lex_sort,
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [0, 1, 1, 0, 0]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [2, -1, 0, -2, 1]),
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [10, 0, 2, -1, 1]),
            num_lex_sort,
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [1, 0, 0, 1, 0]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aAA,bAA,bZ,cC,,eE,"),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aZ,bZ,bBB,,dD,eE,"),
            shortlex_sort,
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", [1, 1, 0, 1, 0, 0]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aZ,bZ,bBB,,dD,eE,"),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aAA,bAA,bZ,cC,,eE,"),
            shortlex_sort,
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", [0, 0, 1, 0, 1, 0]),
        ),
    ],
)
def test_compare_inverse(test_assoc_1, test_assoc_2, sort_key, exp_assoc):
    assert D4M.assoc.assoc_equal(
        test_assoc_1.compare(test_assoc_2, sort_key, inverse=True), exp_assoc
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,sort_key,exp_assoc_1,exp_assoc_2",
    [
        (
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [10, 0, 2, -1, 1]),
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [2, -1, 0, -2, 1]),
            num_lex_sort,
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [1, 0, 0, 1, 0]),
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [0, 1, 1, 0, 0]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [2, -1, 0, -2, 1]),
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [10, 0, 2, -1, 1]),
            num_lex_sort,
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [0, 1, 1, 0, 0]),
            D4M.assoc.Assoc("a,a,b,b,c,", "A,B,A,B,C,", [1, 0, 0, 1, 0]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aAA,bAA,bZ,cC,,eE,"),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aZ,bZ,bBB,,dD,eE,"),
            shortlex_sort,
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", [0, 0, 1, 0, 1, 0]),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", [1, 1, 0, 1, 0, 0]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aZ,bZ,bBB,,dD,eE,"),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aAA,bAA,bZ,cC,,eE,"),
            shortlex_sort,
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", [1, 1, 0, 1, 0, 0]),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", [0, 0, 1, 0, 1, 0]),
        ),
    ],
)
def test_compare_include_inverse(
    test_assoc_1, test_assoc_2, sort_key, exp_assoc_1, exp_assoc_2
):
    compared, inv_compared = test_assoc_1.compare(
        test_assoc_2, sort_key, include_inverse=True
    )
    assert D4M.assoc.assoc_equal(compared, exp_assoc_1)
    assert D4M.assoc.assoc_equal(inv_compared, exp_assoc_2)


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", [1, 2]),
            D4M.assoc.Assoc("a,", "A,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,"),
            D4M.assoc.Assoc([], [], []),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA,bA,bB,"),
            D4M.assoc.Assoc("a,b,c,", "A,B,C,", "aA,bB,cC,"),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
        ),
        (D4M.assoc.Assoc("a,b,", "A,B,", [1, 2]), 1, D4M.assoc.Assoc("a,", "A,", 1)),
    ],
)
def test_eq(test_assoc_1, test_assoc_2, exp_assoc):
    assert D4M.assoc.assoc_equal(
        (test_assoc_1 == test_assoc_2), exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", [1, 2]),
            D4M.assoc.Assoc("b,", "B,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", "aA,bB,"),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", "aA,bA,bB,"),
            D4M.assoc.Assoc("a,b,c,", "A,B,C,", "aA,bB,cC,"),
            D4M.assoc.Assoc("b,c,", "A,C,", 1),
        ),
        (D4M.assoc.Assoc("a,b,", "A,B,", [1, 2]), 1, D4M.assoc.Assoc("b,", "B,", 1)),
    ],
)
def test_ne(test_assoc_1, test_assoc_2, exp_assoc):
    assert D4M.assoc.assoc_equal(
        (test_assoc_1 != test_assoc_2), exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", [1, 1]),
            D4M.assoc.Assoc("a,b,", "A,B,", [1, 2]),
            D4M.assoc.Assoc("a,b,", "A,B,", [0, 1]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [-1, 1, 2]),
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [0, 0, 0]),
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [1, 0, 0]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aA1,,bA1,bB2,"),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aA2,aB2,,bB1,"),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [1, 1, 0, 0]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [10, 0, 2, -2]),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [2, -1, 0, -2]),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [0, 0, 0, 0]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aAA,bAA,bZ,cC,,eE,"),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aZ,bZ,bBB,,dD,eE,"),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", [1, 1, 0, 0, 1, 0]),
        ),
    ],
)
def test_lt(test_assoc_1, test_assoc_2, exp_assoc):
    assert D4M.assoc.assoc_equal(
        test_assoc_1 < test_assoc_2, exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", [1, 1]),
            D4M.assoc.Assoc("a,b,", "A,B,", [1, 2]),
            D4M.assoc.Assoc("a,b,", "A,B,", [0, 0]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [-1, 1, 2]),
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [0, 0, 0]),
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [0, 1, 1]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aA1,,bA1,bB2,"),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aA2,aB2,,bB1,"),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [0, 0, 1, 1]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [10, 0, 2, -2]),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [2, -1, 0, -2]),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [1, 1, 1, 0]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aAA,bAA,bZ,cC,,eE,"),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aZ,bZ,bBB,,dD,eE,"),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", [0, 0, 1, 1, 0, 0]),
        ),
    ],
)
def test_gt(test_assoc_1, test_assoc_2, exp_assoc):
    assert D4M.assoc.assoc_equal(
        test_assoc_1 > test_assoc_2, exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", [1, 1]),
            D4M.assoc.Assoc("a,b,", "A,B,", [1, 2]),
            D4M.assoc.Assoc("a,b,", "A,B,", [1, 1]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [-1, 1, 2]),
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [0, 0, 0]),
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [1, 0, 0]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aA1,,bA1,bB2,"),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aA2,aB2,,bB1,"),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [1, 1, 0, 0]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [10, 0, 2, -2]),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [2, -1, 0, -2]),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [0, 0, 0, 1]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aAA,bAA,bZ,cC,,eE,"),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aZ,bZ,bBB,,dD,eE,"),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", [1, 1, 0, 0, 1, 1]),
        ),
    ],
)
def test_le(test_assoc_1, test_assoc_2, exp_assoc):
    assert D4M.assoc.assoc_equal(
        test_assoc_1 <= test_assoc_2, exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "test_assoc_1,test_assoc_2,exp_assoc",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", [1, 1]),
            D4M.assoc.Assoc("a,b,", "A,B,", [1, 2]),
            D4M.assoc.Assoc("a,b,", "A,B,", [1, 0]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [-1, 1, 2]),
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [0, 0, 0]),
            D4M.assoc.Assoc("a,b,b,", "A,A,B,", [0, 1, 1]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aA1,,bA1,bB2,"),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", "aA2,aB2,,bB1,"),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [0, 0, 1, 1]),
        ),
        (
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [10, 0, 2, -2]),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [2, -1, 0, -2]),
            D4M.assoc.Assoc("a,a,b,b,", "A,B,A,B,", [1, 1, 1, 1]),
        ),
        (
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aAA,bAA,bZ,cC,,eE,"),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", "aZ,bZ,bBB,,dD,eE,"),
            D4M.assoc.Assoc("a,b,b,c,d,e,", "A,A,B,C,D,E,", [0, 0, 1, 1, 0, 1]),
        ),
    ],
)
def test_ge(test_assoc_1, test_assoc_2, exp_assoc):
    assert D4M.assoc.assoc_equal(
        test_assoc_1 >= test_assoc_2, exp_assoc, return_info=True
    )


@pytest.mark.parametrize(
    "row,col,val,split_separator,int_aware,exp",
    [
        (
            "a,b,a,",
            "A,B,B,",
            "1,2,3,",
            None,
            None,
            D4M.assoc.Assoc("a,b,a,", "A|1,B|2,B|3,", 1),
        ),
        (
            "a,b,a,",
            "A,B,B,",
            "1,2,3,",
            None,
            True,
            D4M.assoc.Assoc("a,b,a,", "A|1,B|2,B|3,", 1),
        ),
        (
            "a,b,a,",
            "A,B,B,",
            "1,2,3,",
            None,
            False,
            D4M.assoc.Assoc("a,b,a,", "A|1,B|2,B|3,", 1),
        ),
        (
            "a,b,a,",
            "A,B,B,",
            "aA,bB,aB,",
            None,
            None,
            D4M.assoc.Assoc("a,b,a,", "A|aA,B|bB,B|aB,", 1),
        ),
        (
            "a,b,a,",
            "A,B,B,",
            [1, 2, 3],
            ":",
            None,
            D4M.assoc.Assoc("a,b,a,", "A:1,B:2,B:3,", 1),
        ),
        (
            "a,b,a,",
            "A,B,B,",
            [1, 2, 3],
            ":",
            True,
            D4M.assoc.Assoc("a,b,a,", "A:1,B:2,B:3,", 1),
        ),
        (
            "a,b,a,",
            "A,B,B,",
            [1, 2, 3],
            ":",
            False,
            D4M.assoc.Assoc("a,b,a,", "A:1.0,B:2.0,B:3.0,", 1),
        ),
        (
            "a,b,a,",
            "A,B,B,",
            "1,2,3,",
            "/",
            None,
            D4M.assoc.Assoc("a,b,a,", "A/1,B/2,B/3,", 1),
        ),
    ],
)
def test_val2col(row, col, val, split_separator, int_aware, exp):
    split_separator, int_aware = _replace_default_args(
        D4M.util.catstr, separator=split_separator, int_aware=int_aware
    )
    assoc_ = D4M.assoc.Assoc(row, col, val)
    assert D4M.assoc.assoc_equal(
        D4M.assoc.val2col(assoc_, separator=split_separator, int_aware=int_aware), exp
    )


@pytest.mark.parametrize(
    "A,separator,convert,row,col,val",
    [
        (
            D4M.assoc.Assoc("a,b,a,", "A|1,B|2,B|3,", 1),
            None,
            None,
            "a,b,a,",
            "A,B,B,",
            [1, 2, 3],
        ),
        (
            D4M.assoc.Assoc("a,b,a,", "A|1,B|2,B|3,", 1),
            None,
            False,
            "a,b,a,",
            "A,B,B,",
            ["1", "2", "3"],
        ),
        (
            D4M.assoc.Assoc("a,b,a,", "A|aA,B|bB,B|aB,", 1),
            None,
            None,
            "a,b,a,",
            "A,B,B,",
            "aA,bB,aB,",
        ),
        (
            D4M.assoc.Assoc("a,b,a,", "A:1,B:2,B:3,", 1),
            ":",
            None,
            "a,b,a,",
            "A,B,B,",
            [1, 2, 3],
        ),
        (
            D4M.assoc.Assoc("a,b,a,", "A:1,B:2,B:3,", 1),
            ":",
            False,
            "a,b,a,",
            "A,B,B,",
            ["1", "2", "3"],
        ),
        (D4M.assoc.Assoc([], [], []), None, None, [], [], []),
    ],
)
def test_col_to_type(A, separator, convert, row, col, val):
    separator, convert = _replace_default_args(
        D4M.assoc.col_to_type, separator=separator, convert=convert
    )
    B = D4M.assoc.Assoc(row, col, val)
    assert D4M.assoc.assoc_equal(
        D4M.assoc.col_to_type(A, separator=separator, convert=convert),
        B,
        return_info=True,
    )


@pytest.mark.parametrize(
    "assoc1,assoc2,equal",
    [
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", [1, 1]),
            True,
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,", "A,B,", [1.0, 1.0]),
            True,
        ),
        (
            D4M.assoc.Assoc(["a", "b"], ["A", "B"], 1),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            True,
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            D4M.assoc.Assoc("a,b,a,", "A,B,B,", [1, 1, 1]),
            False,
        ),
        (
            D4M.assoc.Assoc("a,b,", "A,B,", [1, 2]),
            D4M.assoc.Assoc("a,b,c,", "A,B,C,", [1, 2, 0]),
            True,
        ),
        (D4M.assoc.Assoc("b,a,", "B,A,", 1), D4M.assoc.Assoc("a,b,", "A,B,", 1), True),
        (
            D4M.assoc.Assoc("a,b,a,", "A,B,B,", [1, 1, 0]).condense(),
            D4M.assoc.Assoc("a,b,", "A,B,", 1),
            True,
        ),
    ],
)
def test_assoc_equal(assoc1, assoc2, equal):
    assert D4M.assoc.assoc_equal(assoc1, assoc2, return_info=True) == equal


# TODO?: def test_readcsvtotriples
# TODO?: def test_readcsv
# TODO?: def test_writecsv
