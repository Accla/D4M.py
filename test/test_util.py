import pytest
import numpy as np

import D4M.util
from D4M.util import replace_default_args

# TODO: def test_string_gen
# TODO: def test_num_string_gen


@pytest.mark.parametrize(
    "test,exp", [(1.0, True), (-1, True), ("a", False), ("1", False), ("a1", False)]
)
def test_is_numeric(test, exp):
    assert D4M.util.is_numeric(test) == exp


@pytest.mark.parametrize(
    "test1,test2,returnbool,exp,returnexp1,returnexp2",
    [
        (
            np.array([0, 1, 4, 6]),
            np.array([0, 4, 7]),
            True,
            np.array([0, 1, 4, 6, 7]),
            np.array([0, 1, 2, 3]),
            np.array([0, 2, 4]),
        ),
        (
            np.array([0, 1, 4, 6]),
            np.array([0, 4, 7]),
            False,
            np.array([0, 1, 4, 6, 7]),
            None,
            None,
        ),
    ],
)
def test_sorted_union(test1, test2, returnbool, exp, returnexp1, returnexp2):
    if returnbool:
        union, index_map_1, index_map_2 = D4M.util.sorted_union(
            test1, test2, return_index=True
        )
        assert np.array_equal(union, exp)
        assert np.array_equal(index_map_1, returnexp1)
        assert np.array_equal(index_map_2, returnexp2)
    else:
        union = D4M.util.sorted_union(test1, test2)
        assert np.array_equal(union, exp)


@pytest.mark.parametrize(
    "test1,test2,returnbool,exp,returnexp1,returnexp2",
    [
        (
            np.array([0, 1, 4]),
            np.array([0, 4, 7]),
            True,
            np.array([0, 4]),
            np.array([0, 2]),
            np.array([0, 1]),
        ),
        (np.array([0, 1, 4]), np.array([0, 4, 7]), False, np.array([0, 4]), None, None),
    ],
)
def test_sorted_intersect(test1, test2, returnbool, exp, returnexp1, returnexp2):
    if returnbool:
        intersection, index_map_1, index_map_2 = D4M.util.sorted_intersect(
            test1, test2, return_index=True
        )
        assert np.array_equal(intersection, exp)
        assert np.array_equal(index_map_1, returnexp1)
        assert np.array_equal(index_map_2, returnexp2)
    else:
        intersection = D4M.util.sorted_intersect(test1, test2)
        assert np.array_equal(intersection, exp)


# TODO: def test_select_items


@pytest.mark.parametrize(
    "test,testsub,exp",
    [
        ("aa,bb,ab,", "a,", [0, 2]),
        ("aa,bb,ab,", "b,", [1, 2]),
        ("aa,bb,ab,", "c,", []),
        ("aa,bb,ab,", "a,b,", [0, 1, 2]),
        (["aa", "bb", "ab"], "a,", [0, 2]),
        (["aa", "bb", "ab"], "b,", [1, 2]),
        (["aa", "bb", "ab"], "c,", []),
        (["aa", "bb", "ab"], "a,b,", [0, 1, 2]),
        (["aa", "bb", "ab"], ["a"], [0, 2]),
        (["aa", "bb", "ab"], ["b"], [1, 2]),
        (["aa", "bb", "ab"], ["c"], []),
        (["aa", "bb", "ab"], ["a", "b"], [0, 1, 2]),
        ("aa,bb,ab,", ["a"], [0, 2]),
        ("aa,bb,ab,", ["b"], [1, 2]),
        ("aa,bb,ab,", ["c"], []),
        ("aa,bb,ab,", ["a", "b"], [0, 1, 2]),
    ],
)
def test_contains(test, testsub, exp):
    assert D4M.util.contains(testsub)(test) == exp


@pytest.mark.parametrize(
    "test,testsub,exp",
    [
        ("aa,bb,ab,", "a,", [0, 2]),
        ("aa,bb,ab,", "b,", [1]),
        ("aa,bb,ab,", "c,", []),
        ("aa,bb,ab,", "a,b,", [0, 1, 2]),
        (["aa", "bb", "ab"], "a,", [0, 2]),
        (["aa", "bb", "ab"], "b,", [1]),
        (["aa", "bb", "ab"], "c,", []),
        (["aa", "bb", "ab"], "a,b,", [0, 1, 2]),
        (["aa", "bb", "ab"], ["a"], [0, 2]),
        (["aa", "bb", "ab"], ["b"], [1]),
        (["aa", "bb", "ab"], ["c"], []),
        (["aa", "bb", "ab"], ["a", "b"], [0, 1, 2]),
        ("aa,bb,ab,", ["a"], [0, 2]),
        ("aa,bb,ab,", ["b"], [1]),
        ("aa,bb,ab,", ["c"], []),
        ("aa,bb,ab,", ["a", "b"], [0, 1, 2]),
    ],
)
def test_startswith(test, testsub, exp):
    assert D4M.util.startswith(testsub)(test) == exp


@pytest.mark.parametrize("obj,exp", [("1", 1), ("1.2", 1.2), ("-5", -5)])
def test_str_to_num(obj, exp):
    assert D4M.util.str_to_num(obj) == exp


@pytest.mark.parametrize(
    "test_word,suffix,exp_word",
    [
        ("1.0", ".0", "1"),
        ("1.1", ".0", "1.1"),
        ("abcd", "c", "abcd"),
        ("abcd", "d", "abc"),
    ],
)
def test_remove_suffix(test_word, suffix, exp_word):
    assert D4M.util.remove_suffix(test_word, suffix) == exp_word


# TODO: def test_remove_suffix


@pytest.mark.parametrize(
    "obj,exp",
    [
        ([0, 1], ["0", "1"]),
        ([0, -1], ["0", "-1"]),
        ([0, 1, 0.12, -1], ["0", "1", "0.12", "-1"]),
    ],
)
def test_num_to_str(obj, exp):
    arr = np.array(obj)
    strarr = np.array(exp)
    assert np.array_equal(D4M.util.num_to_str(arr), strarr)


@pytest.mark.parametrize(
    "test_object,can_san",
    [
        ([0, 1, 2], True),
        (0, True),
        ("ab", True),
        ({0, 1, 2}, False),
        ({0: "a", 1: "b", 2: "c"}, False),
    ],
)
def test_can_sanitize(test_object, can_san):
    assert D4M.util.can_sanitize(test_object) == can_san


@pytest.mark.parametrize(
    "test,prevent_upcasting,exp",
    [
        ([1, 1], False, np.array([1, 1], dtype=int)),
        ([1, 2.5], True, np.array([1, 2.5], dtype=object)),
        ([1, 2.5], False, np.array([1.0, 2.5], dtype=float)),
        ([1, 2.5], False, np.array([1.0, 2.5], dtype=float)),
        ([1, 2.5], True, np.array([1, 2.5], dtype=object)),
        (1, False, np.array([1], dtype=int)),
        ("a,b,", False, np.array(["a", "b"], dtype=str)),
        ("1,1,", False, np.array(["1", "1"], dtype=str)),
        ("", False, np.array([""])),
    ],
)
def test_sanitize(test, prevent_upcasting, exp):
    assert np.array_equal(
        exp,
        D4M.util.sanitize(test, prevent_upcasting=prevent_upcasting),
    )


@pytest.mark.parametrize(
    "test,exp",
    [
        ("1,1,", np.array([1, 1], dtype=int)),
        ("1", np.array([1], dtype=int)),
        ("1.0", np.array([1], dtype=int)),
        ("1.0,2,", np.array([1, 2], dtype=int)),
        ("", np.array([0], dtype=int)),
        ("3.5", np.array([3.5], dtype=float)),
        ("2.5,-1.4,", np.array([2.5, -1.4], dtype=float)),
    ],
)
def test_str_to_num(test, exp):
    assert np.array_equal(D4M.util.str_to_num(test), exp)


@pytest.mark.parametrize(
    "test,exp",
    [
        ("1,1,", np.array([1, 1], dtype=int)),
        ("1.0,2.7,", np.array([1.0, 2.7], dtype=float)),
        ("a,b,", np.array(["a", "b"], dtype=str)),
        ("a,1,", np.array(["a", 1], dtype=object)),
        ("a,1.0,", np.array(["a", 1.0], dtype=object)),
    ],
)
def test_str_to_num_silent(test, exp):
    assert np.array_equal(D4M.util.str_to_num(test, silent=True), exp)


@pytest.mark.parametrize(
    "test",
    [
        "a,b,",
        "a,1,",
        "1.0,a,-2,",
    ],
)
def test_str_to_num_loud(test):
    with pytest.raises(ValueError):
        D4M.util.str_to_num(test, silent=False)


# TODO: def test_to_db_string
# TODO: def test_np_sorted


@pytest.mark.parametrize(
    "row,col,val,func,agg_row,agg_col,agg_val",
    [
        (
            ["a", "a", "b"],
            ["A", "A", "B"],
            [1, 2, 3],
            D4M.util.add,
            ["a", "b"],
            ["A", "B"],
            [3, 3],
        ),
        (
            ["a", "a", "b"],
            ["A", "A", "B"],
            [1, 2, 3],
            D4M.util.first,
            ["a", "b"],
            ["A", "B"],
            [1, 3],
        ),
        (
            ["a", "a", "b"],
            ["A", "A", "B"],
            [1, 2, 3],
            D4M.util.last,
            ["a", "b"],
            ["A", "B"],
            [2, 3],
        ),
        (
            ["a", "a", "b"],
            ["A", "A", "B"],
            [2, 2, 3],
            D4M.util.times,
            ["a", "b"],
            ["A", "B"],
            [4, 3],
        ),
        (
            ["a", "a", "a", "b"],
            ["A", "A", "A", "B"],
            [1, 2, 0, 3],
            min,
            ["a", "b"],
            ["A", "B"],
            [0, 3],
        ),
    ],
)
def test_aggregate_triples(row, col, val, func, agg_row, agg_col, agg_val):
    agg_row = np.array(agg_row)
    agg_col = np.array(agg_col)
    agg_val = np.array(agg_val)

    new_row, new_col, new_val = D4M.util.aggregate_triples(row, col, val, func)
    assert np.array_equal(new_row, agg_row)
    assert np.array_equal(new_col, agg_col)
    assert np.array_equal(new_val, agg_val)


@pytest.mark.parametrize(
    "new_entry,sorted_array,array_of_indices,exp_sorted_array,exp_array_of_indices",
    [
        (
            "aaB",
            np.array(["aA", "aB", "bB"]),
            np.array([0, 1, 1, 0, 2]),
            np.array(["aA", "aB", "aaB", "bB"]),
            np.array([0, 1, 1, 0, 3, 2]),
        ),
        (
            "c",
            np.array(["aA", "aB", "bB"]),
            np.array([0, 1, 1, 0, 2]),
            np.array(["aA", "aB", "bB", "c"]),
            np.array([0, 1, 1, 0, 2, 3]),
        ),
    ],
)
def test_sorted_append(
    new_entry, sorted_array, array_of_indices, exp_sorted_array, exp_array_of_indices
):
    new_sorted_array, new_array_of_indices = D4M.util.sorted_append(
        new_entry, sorted_array, array_of_indices
    )
    assert np.array_equal(new_sorted_array, exp_sorted_array)
    assert np.array_equal(new_array_of_indices, exp_array_of_indices)


@pytest.mark.parametrize(
    "array_of_indices,sorted_bad_indices,size,offset,mark,exp_array",
    [
        (np.array([0, 1, 2, 3]), [2], 4, None, -1, np.array([0, 1, -1, 2])),
        (np.array([0, 1, 2, 3]), [2], 4, None, None, np.array([0, 1, 2])),
        (np.array([0, 2, 4, 6]), [1], 7, None, -1, np.array([0, 1, 3, 5])),
        (np.array([0, 2, 4, 6]), [1], 7, None, None, np.array([0, 1, 3, 5])),
        (np.array([0, 2, 4, 6]), [1, 3], 7, None, -1, np.array([0, 1, 2, 4])),
        (np.array([0, 2, 4, 6]), [1, 3], 7, None, None, np.array([0, 1, 2, 4])),
        (np.array([1, 2, 3, 4]), [2], 4, 1, 0, np.array([1, 2, 0, 3])),
        (np.array([1, 2, 3, 4]), [2], 4, 1, None, np.array([1, 2, 3])),
        (np.array([1, 3, 5, 7]), [1], 7, 1, 0, np.array([1, 2, 4, 6])),
        (np.array([-1, 1, 3, 5]), [1, 3], 7, -1, -2, np.array([-1, 0, 1, 3])),
        (np.array([3, 2, 5, 1]), [1, 3], 5, 1, 0, np.array([2, 0, 3, 1])),
        (np.array([]), [1, 3], 5, 1, 0, np.array([])),
    ],
)
def test_update_indices(
    array_of_indices, sorted_bad_indices, size, offset, mark, exp_array
):
    offset, mark = replace_default_args(
        D4M.util.update_indices, offset=offset, mark=mark
    )
    updated_array = D4M.util.update_indices(
        array_of_indices, sorted_bad_indices, size, offset=offset, mark=mark
    )
    assert np.array_equal(updated_array, exp_array)


@pytest.mark.parametrize(
    "s1,s2,sep,exp",
    [
        (np.array(["a", "b"]), np.array(["A", "B"]), None, np.array(["a|A", "b|B"])),
        (np.array(["a", "b"]), np.array(["A", "B"]), ":", np.array(["a:A", "b:B"])),
        (np.array([1, 1]), np.array(["A", "B"]), None, np.array(["1|A", "1|B"])),
    ],
)
def test_catstr(s1, s2, sep, exp):
    sep = replace_default_args(D4M.util.catstr, separator=sep)
    assert np.array_equal(exp, D4M.util.catstr(s1, s2, separator=sep))


# TODO: def test_get_default_args?
# TODO: def test_replace_default_args?


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
