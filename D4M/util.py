# Import packages
import numpy as np
import random
import string
import operator
import warnings
from numbers import Number
from inspect import signature
from typing import Any, Union, Tuple, Optional, Callable, Sequence, List, Dict
from collections.abc import Sequence as SequenceLike

# Use List & Dict for backwards (<3.9) compatibility

KeyVal = Union[str, Number]
StrList = Union[str, Sequence[str]]
ArrayLike = Union[KeyVal, Sequence[KeyVal], np.ndarray]
Selectable = Union[ArrayLike, slice, Callable]


# Auxiliary/Helper functions


def string_gen(length_: int) -> str:
    """Create randomly-generated string of given length."""
    rand_string = ""
    for _ in range(length_):
        rand_string += random.SystemRandom().choice(string.ascii_letters)
    return rand_string


def num_string_gen(length_: int, upper_bound: int) -> str:
    """Create string list of integers <= upper_bound of given length."""
    rand_string = [str(random.randint(0, upper_bound)) for _ in range(length_)] + [""]
    rand_string = ",".join(rand_string)
    return rand_string


def is_numeric(object_: Any) -> bool:
    """Check if object_ is numeric (int, float, complex, etc) or not."""
    return isinstance(object_, Number)


def sorted_union(
    array_1: np.ndarray, array_2: np.ndarray, return_index: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return the union of two sorted numpy arrays with index maps (if return_index=True).
    Usage:
        union = sorted_union(array_1, array_2)
    Input:
        array_1 = sorted numpy array of values with no duplicates
        array_2 = sorted numpy array of values with no duplicates
        return_index = boolean
    Output:
        union = sorted array of values coming from either array_1 or array_2
        index_map_1 = list of indices of the elements of array_1 in union
        index_map_2 = list of indices of the elements of array_2 in union
    Example:
        sorted_union(np.array([0,1,4,6]), np.array([0,4,7]), return_index = True)
            = np.array([0,1,4,6,7]), [0,1,2,3], [0,2,4]
        sorted_union(np.array([0,1,4,6]), np.array([0,4,7]))
            = np.array([0,1,4,6,7])
    """
    union = list()
    index_map_1, index_map_2 = list(), list()
    index_1, index_2 = 0, 0

    size_1, size_2 = len(array_1), len(array_2)
    union_size = 0

    while index_1 < size_1 or index_2 < size_2:
        if index_1 >= size_1:
            if return_index:
                index_map_2.append(union_size)
            union.append(array_2[index_2])
            union_size += 1
            index_2 += 1
        elif index_2 >= size_2:
            if return_index:
                index_map_1.append(union_size)
            union.append(array_1[index_1])
            union_size += 1
            index_1 += 1
        elif array_1[index_1] == array_2[index_2]:
            if return_index:
                index_map_1.append(union_size)
                index_map_2.append(union_size)
            union.append(array_1[index_1])
            union_size += 1
            index_1 += 1
            index_2 += 1
        elif array_1[index_1] < array_2[index_2]:
            if return_index:
                index_map_1.append(union_size)
            union.append(array_1[index_1])
            union_size += 1
            index_1 += 1
        else:
            if return_index:
                index_map_2.append(union_size)
            union.append(array_2[index_2])
            union_size += 1
            index_2 += 1

    union = np.array(union)

    if return_index:
        index_map_1 = np.array(index_map_1)
        index_map_2 = np.array(index_map_2)
        return union, index_map_1, index_map_2
    else:
        return union


def sorted_intersect(
    array_1: np.ndarray,
    array_2: np.ndarray,
    return_index: bool = False,
    return_index_1: bool = False,
    return_index_2: bool = False,
) -> Union[
    np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """Return the intersection of two sorted numpy arrays with index maps
    (if return_index, return_index_1, or return_index_2 are True).
    Usage:
        intersection = sorted_intersection(array_1, array_2)
    Input:
        array_1 = sorted numpy array of values with no duplicates
        array_2 = sorted numpy array of values with no duplicates
        return_index = (Optional, default False) Boolean indicating whether index maps from the intersection
            into array_1 and array_2 should be returned. return_index_1 and return_index_2 are the same, but only
            apply to index maps into array_1 and array_2, respectively. If return_index is True, then return_index_1
            and return_index_2 are assumed True.
    Output:
        intersection = sorted array of values coming from both array_1 and array_2
        index_map_1 = list of indices of elements of intersection in array_1
        index_map_2 = list of indices of elements of intersection in array_2
    Example:
        sorted_intersect(np.array([0,1,4]), np.array([0,4,7]), return_index = True)
            = np.array([0,4]), [0,2], [0,1]
        sorted_intersect(np.array([0,1,4]), np.array([0,4,7]))
            = np.array([0,4])
    """
    set_2 = set(array_2)
    intersection = [item for item in array_1 if item in set_2]

    if return_index or return_index_1:
        array_1_index = {array_1[index]: index for index in range(len(array_1))}
        index_map_1 = [array_1_index[x] for x in intersection]
    else:
        index_map_1 = None

    if return_index or return_index_2:
        array_2_index = {array_2[index]: index for index in range(len(array_2))}
        index_map_2 = [array_2_index[x] for x in intersection]
    else:
        index_map_2 = None

    if return_index:
        return np.array(intersection), np.array(index_map_1), np.array(index_map_2)
    elif return_index_1:
        return np.array(intersection), np.array(index_map_1)
    elif return_index_2:
        return np.array(intersection), np.array(index_map_2)
    else:
        return np.array(intersection)


def select_items(
    selection: Selectable, keys: ArrayLike, return_indices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Select a subsequence from the sequence keys using given selection.
    Inputs:
        selection = a callable compatible with input keys and returning a list of indices of keys,
            or a slice object,
            or a string, a sequence of strings, a sequence of ints, a sequence of bools, or a sequence of floats
        keys = a sorted numpy array with no duplicates
    Output:
        select_items(selection, keys) = subsequence, where
            - if selection is a callable, then subsequence = keys[selection(keys)]
            - if selection is a slice object, with subsequence = keys[object_]
            - if, after being sanitized, selection is an array of strings containing ':'...
                ...and is of the form np.array([initial, ':', final]), then subsequence = keys[in: fin], where
                    in = least integer such that initial <= keys[in] and
                    out = greatest integer such that keys[fin] <= final, or...
                ...and is of the form np.array([initial, ':']), then subsequence = keys[in:], where
                    in = least integer such that initial <= keys[in], or...
                ...and is of the form np.array([':', final]), then subsequence = keys[:fin], where
                    fin = greatest integer such that keys[fin] <= final, or...
                ...and is of the form np.array([':']), then selection = keys, or...
                ...otherwise raises a ValueError
            - if, after being sanitized, selection is an array of integers or bools, then
                subsequence = keys[selection]
            - otherwise, selection is sanitized and subsequence = selection
        select_items(selection, keys, return_indices=True) = subsequence, index_map, where
            subsequence is as above and index_map is a numpy array of ints such that subsequence = keys[index_map]
    Note:
        - Unlike slice objects, string slices with an upper limit (array of strings of the form
            [initial, ':', final] or [':', final]) ARE inclusive on the right, if final is in keys
    """
    colon = np.array([":"])
    keys = sanitize(keys)

    if callable(selection):
        # If function on iterables returning a list of indices, apply to keys and get subarray
        selection = selection(keys)
        selection = np.unique(selection).astype(int)
        clean_selection = keys[selection]
        index_map = selection
    elif isinstance(selection, slice):
        # If slice object, take that slice of keys
        clean_selection = keys[selection]
        index_map = np.arange(0, len(keys))[selection]
    else:
        # Otherwise, sanitize and find any instances of the string ':'
        selection = sanitize(selection)
        # Check if ':' is compatible with selection.dtype to make numpy happy, then test membership if so
        if np.issubdtype(colon.dtype, selection.dtype) and np.isin(":", selection):
            colon_indices = np.nonzero(selection == ":")[0]
        else:
            colon_indices = np.empty(0, dtype=int)
        if selection.dtype == int or selection.dtype == bool:
            # If array of ints, treat as indices of keys
            # If array of Booleans, extract indices where True and then treat as array of ints
            if selection.dtype == bool:
                selection = np.nonzero(selection)[0]  # Already unique & sorted
            else:
                selection = np.unique(selection)
            clean_selection = keys[selection]
            index_map = selection
        elif len(colon_indices) == 1 and (
            len(selection) == 1
            or len(selection) == 2
            or (len(selection) == 3 and selection[1] == ":")
        ):
            # If well-formed string slice, take that slice of keys
            if len(selection) == 1:
                start_compare = np.ones(len(keys))
                stop_compare = np.zeros(len(keys))
            elif len(selection) == 2 and selection[0] == ":":
                start_compare = np.ones(len(keys))
                stop_compare = keys > selection[1]
            elif len(selection) == 2 and selection[1] == ":":
                start_compare = keys >= selection[0]
                stop_compare = np.zeros(len(keys))
            else:
                start_compare = keys >= selection[0]
                stop_compare = keys > selection[2]

            try:
                start_index = np.nonzero(start_compare)[0][0]
            except IndexError:
                start_index = 0
            try:
                stop_index = np.nonzero(stop_compare)[0][0]
            except IndexError:
                stop_index = len(keys)

            clean_selection = keys[start_index:stop_index]
            index_map = np.arange(start_index, stop_index)
        elif np.isin(":", selection):
            # If contains ':' but not dealt with, ill-formed string slice
            raise ValueError("Improper string slice provided.")
        else:
            # Assume that the elements of selection are actual elements of keys; take intersection
            selection = np.unique(selection)
            clean_selection, selection_map, keys_index_map = np.intersect1d(
                selection, keys, assume_unique=True, return_indices=True
            )
            index_map = keys_index_map

    if return_indices:
        return clean_selection, index_map
    else:
        return clean_selection


def contains(substrings: StrList) -> Callable[[StrList], List[int]]:
    """Return callable which accepts a list of strings and returns the list of indices
    of those strings which contain some element of substrings.
    Usage:
        contains("a,b,")
        contains(['a','b'])
    Inputs:
        substrings = string of (delimiter separated) values (delimiter is last character)
            or list of values of length n
    Outputs:
        func(string_list) = returns a list of indices of the strings in string_list which have some element of
            substrings as a substring
    """
    presanitized = substrings
    substrings = sanitize(substrings)

    def func(
            string_list: Union[StrList, None],
            function_name: str = "contains",
            currying_parameter: StrList = presanitized
    ):
        if string_list is None:
            return function_name, currying_parameter

        string_list = sanitize(string_list)
        good_string_list = list()
        for index in range(len(string_list)):
            item = str(string_list[index])
            for substring in substrings:
                if substring in item:
                    good_string_list.append(index)
                    break
        return good_string_list

    func.__name__ = "contains(" + str(presanitized) + ")"

    return func


def startswith(prefixes: StrList) -> Callable[[StrList], List[int]]:
    """Return callable which accepts a list of strings and returns the list of indices
    of those strings which have some element of prefixes as a prefix.
    Usage:
        startswith("a,b,")
        startswith(['a','b'])
    Inputs:
        prefixes = string of (delimiter separated) values (delimiter is last character)
            or list of values of length n
    Outputs:
        func(string_list) = returns a list of indices of the strings in string_list which have some element of
            prefixes as a prefix
    """
    presanitized = prefixes
    prefixes = sanitize(prefixes)

    def func(
            string_list: Union[StrList, None],
            function_name: str = "startswith",
            currying_parameter: StrList = presanitized
    ):
        if string_list is None:
            return function_name, currying_parameter
        else:
            string_list = sanitize(string_list)
            good_string_list = list()

            for index in range(len(string_list)):
                item = str(string_list[index])
                for prefix in prefixes:
                    if item.startswith(prefix):
                        good_string_list.append(index)
                        break
            return good_string_list

    func.__name__ = "startswith(" + str(presanitized) + ")"

    return func


def _single_str_to_num(
    word: Union[str, Number], silent: bool = False
) -> Union[str, Number]:
    if isinstance(word, str):
        if len(word) == 0:
            # Convert empty string to 0
            num = 0
        else:
            # Check if can be cast as an integer or float
            try:
                num_float = float(word)
                num_int = int(num_float)
                if num_int == num_float:
                    num = num_int
                else:
                    num = num_float
            except ValueError:
                if silent:
                    num = word
                else:
                    raise ValueError(
                        str(word) + " cannot be converted to a valid number."
                    )
    else:
        assert isinstance(word, Number)
        num = word
    return num


def str_to_num(array: ArrayLike, silent: bool = False) -> np.ndarray:
    """Convert string to float if possible."""
    if isinstance(array, str):
        # Check if the array/string is wholly numerical
        try:
            num_array = [_single_str_to_num(array)]
        except ValueError:
            split_array = array.split(array[-1])
            split_array.pop()
            num_array = [
                _single_str_to_num(word, silent=silent) for word in split_array
            ]
    elif isinstance(array, Number):
        num_array = [array]
    else:
        assert hasattr(array, "__iter__")
        num_array = [_single_str_to_num(word, silent=silent) for word in array]

    if not silent:
        num_array = np.array(num_array)
    else:
        num_array_attempt = np.array(num_array)
        if np.issubdtype(num_array_attempt.dtype, int) or np.issubdtype(
            num_array_attempt.dtype, float
        ):
            num_array = num_array_attempt
        else:
            num_array = np.array(num_array, dtype=object)
    return num_array


def remove_suffix(word, suffix):
    if len(word) == 0:
        return word
    elif len(suffix) <= len(word) and word[(-len(suffix)):] == suffix:
        return word[0: (-len(suffix))]
    else:
        return word


def num_to_str(array: ArrayLike, int_aware: bool = True) -> np.ndarray:
    """Convert array of numbers to array of strings."""
    array = sanitize(array)
    stringified_array = array.astype("str")
    if int_aware:
        stringified_array = np.array([remove_suffix(word, ".0") for word in stringified_array])
    return stringified_array


def can_sanitize(object_: Any) -> bool:
    return (
        isinstance(object_, str)
        or isinstance(object_, Number)
        or isinstance(object_, SequenceLike)
        or (
            isinstance(object_, np.ndarray)
            and object_.ndim == 1
            and (
                np.issubdtype(object_.dtype, float)
                or np.issubdtype(object_.dtype, int)
                or np.issubdtype(object_.dtype, str)
            )
        )
    )


def sanitize(
    object_: ArrayLike,
    prevent_upcasting: bool = False,
) -> np.ndarray:
    """Convert
    * strings of (delimiter-separated) values into a numpy array of values (delimiter = last character),
    * iterables into numpy arrays, and
    * all other objects into a numpy array having that object.
    Usage:
        sanitized list = sanitize(obj)
    Inputs:
        object_ = string of (delimiter separated) values (delimiter is last character)
            or iterable of values of length n or single value
        prevent_upcasting = (Optional, default False) Boolean indicating whether potential upcasting when forming
            numpy arrays should be avoided when possible
    Outputs:
        list of values
    Examples:
        sanitize("a,b,") = np.array(['a', 'b'])
        sanitize("1,1,") = np.array(['1', '1'])
        sanitize([10, 3]) = np.array([10, 3])
        sanitize([10, 3.5]) = np.array([10, 3.5], dtype=float)
        sanitize([10, 3.5], prevent_upcasting=True) = np.array([10, 3.5], dtype=object)
        sanitize(1) = np.array([1])
    """
    # If a single string character, treat as single value instead of delimiter
    if isinstance(object_, str):
        if len(object_) == 0:
            return np.array([""])
        elif len(object_) == 1:
            return np.array([object_])
        else:
            # Convert delimiter-separated string list by splitting using last character
            delimiter = object_[-1]
            object_ = object_.split(delimiter)
            object_.pop()  # Get rid of empty strings

    # Convert to numpy array
    if not isinstance(object_, np.ndarray):
        if hasattr(object_, "__iter__"):
            # Only make dtype=object if necessary
            if prevent_upcasting and len({type(item) for item in object_}) > 1:
                object_ = np.array(object_, dtype=object)
            else:
                object_ = np.array(object_)  # Possible silent upcasting
        else:
            object_ = np.array([object_])

    return object_


def to_db_string(object_) -> str:
    """Convert input (either a delimiter-separated string or an iterable) to Accumulo-friendly string."""
    if isinstance(object_, str):
        db_string = object_.replace(object_[-1], "\n")
    elif callable(object_) and object_.__name__.startswith("startswith"):
        args_dict = get_default_args(object_, "function_name", "currying_parameter")
        prefixes = args_dict["currying_parameter"]
        if isinstance(prefixes, str):
            delimiter = prefixes[-1]
        else:
            delimiter = "\n"
        pre_db_string = sanitize(prefixes).astype(str)
        db_string = ""
        for index in range(len(pre_db_string)):
            prefix = pre_db_string[index]
            db_string += prefix + delimiter + ":" + delimiter + prefix + chr(127) + delimiter
    else:
        db_string = sanitize(object_).astype(str)
        db_string = "\n".join(db_string) + "\n"
    return db_string


def from_db_string(db_string: str) -> np.ndarray:
    """Convert db-formatted string to numpy array."""
    return sanitize(db_string.split("\n")[0:-1])


def np_sorted(arr: np.ndarray) -> bool:
    """Determines if given numpy array is sorted."""
    return np.all(arr[:-1] <= arr[1:])


def aggregate_triples(
    row: Sequence,
    col: Sequence,
    val: Sequence,
    func: Callable[[KeyVal, KeyVal], KeyVal],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate (row[i], col[i], val[i]) triples using func as collision function.
    Usage:
        aggregate_triples(row, col, val, func)
    Inputs:
        row = numpy array of length n
        col = numpy array of length n
        val = numpy array of length n
        func = collision function (e.g. add, times, max, min, first, last)
    Output:
        new_row, new_col, new_val = subarrays of row, col, val in which pairs (r, c) = (new_row[i], new_col[i])
            are unique and new_val[i] is the resulting of iteratively applying func to the values corresponding to
            triples (r, c, value) = (row[j], col[j], val[j])
    Example:
        aggregate_triples(['a', 'a', 'b'], ['A', 'A', 'B'], [1, 2, 3], add) = ['a', 'b'], ['A', 'B'], [3, 3]
        aggregate_triples(['a', 'a', 'b'], ['A', 'A', 'B'], [1, 2, 3], first) = ['a', 'b'], ['A', 'B'], [1, 3]
        aggregate_triples(['a', 'a', 'b'], ['A', 'A', 'B'], [1, 2, 3], last) = ['a', 'b'], ['A', 'B'], [2, 3]
        aggregate_triples(['a', 'a', 'a', 'b'], ['A', 'A', 'A', 'B'], [1, 2, 0, 3], min)
            = ['a', 'b'], ['A', 'B'], [0, 3]
        (where lists are stand-ins for the corresponding numpy arrays)
    """
    aggregate_dict = dict()
    for index in range(len(row)):
        if (row[index], col[index]) not in aggregate_dict:
            aggregate_dict[(row[index], col[index])] = val[index]
        else:
            aggregate_dict[(row[index], col[index])] = func(
                aggregate_dict[(row[index], col[index])], val[index]
            )

    new_row = np.array([item[0] for item in list(aggregate_dict.keys())])
    new_col = np.array([item[1] for item in list(aggregate_dict.keys())])
    new_val = np.array(list(aggregate_dict.values()))
    return new_row, new_col, new_val


def sorted_append(
    new_entry: KeyVal,
    sorted_array: np.ndarray,
    array_of_indices: np.ndarray,
    new_entry_name: str = "new_entry",
    sorted_array_name: str = "sorted_array",
) -> Tuple[np.ndarray, np.ndarray]:
    """Add new entry to a sorted array, sort, and update an array of indices to reflect new entry.
    Inputs:
        new_entry = value to add to given sorted_array
        sorted_array = sorted, unique numpy array
        array_of_indices = numpy array whose entries are indices in reference to sorted_array
        new_entry_name = (Optional, default 'new_entry') how new_entry argument is referred to in warnings
        sorted_array_name = (Optional, default 'sorted_array') how sorted_array argument is referred to in warnings
    Outputs:
        updated_sorted = sorted, unique union of sorted_array and np.array([new_entry])
        updated_indices = update of array_of_indices with indices updated to refer to corresponding entries of
            updated_sorted coming from sorted_unique, additionally appended with the index of new_entry in
            updated_sorted
    Notes:
        - If new_entry is already present in sorted_array, updated_indices is array_of_indices appended with the
            index of new_entry in sorted_array.
        - If the datatype of new_entry is incompatible with the datatype of entries of sorted_array (in particular,
            <= comparisons cannot be made between new_entry and entries of sorted_array), then entries of
            sorted_array are upcast to be compatible, if possible, and a warning is raised.
        - If new_entry and the entries of sorted_array are numerical, then they will silently be upcast to the
            least general datatype of them.
    """
    try:
        compare = sorted_array >= new_entry
        if np.any(compare):
            new_index = np.nonzero(compare)[0][0]
            if sorted_array[new_index] != new_entry:
                updated_sorted = np.append(
                    np.append(sorted_array[0:new_index], np.array([new_entry])),
                    sorted_array[new_index:],
                )
                offset_array = np.append(
                    np.arange(new_index),
                    np.arange(new_index + 1, len(sorted_array) + 1),
                )
                updated_indices = offset_array[array_of_indices].astype(
                    int
                )  # Re-index existing entries
            else:
                updated_sorted = sorted_array
                updated_indices = array_of_indices
            updated_indices = np.append(
                updated_indices, new_index
            )  # Append index of new_entry
        else:
            # Otherwise, row_key is after all entries in sorted_array
            updated_sorted = np.append(sorted_array, new_entry)
            updated_indices = np.append(array_of_indices, len(sorted_array))
    except TypeError:
        # Assume issue is that >= comparison fails due to incompatible datatypes
        warnings.warn(
            "Incompatible dtypes; the dtypes of "
            + sorted_array_name
            + " may be upcast to make it "
            + "compatible with given "
            + new_entry_name
            + "."
        )
        # Try forcing an upcast by appending new_entry to sorted_array anyway
        updated_presorted = np.append(sorted_array, new_entry)
        updated_sorted, inverse_map = np.unique(updated_presorted, return_inverse=True)
        updated_presorted_indices = np.append(array_of_indices, len(sorted_array))
        updated_indices = inverse_map[updated_presorted_indices]

    return updated_sorted, updated_indices


def update_indices(
    array_of_indices: np.ndarray,
    sorted_bad_indices: List[int],
    size: int,
    offset: int = 0,
    mark: Optional[int] = None,
) -> np.ndarray:
    """Given a numpy array of indices of an understood sequence of values and a list of indices of that sequence which
    are to be removed, update array of indices to be with respect to post-deletion sequence.
    Inputs:
        array_of_indices = numpy array of indices (with possible offset) of an understood sequence
        sorted_bad_indices = sorted list of indices of understood sequence to be deleted/removed
        size = length/size of understood sequence (i.e., must be at least as large as the largest index encountered
            + 1)
        offset = (Optional, default 0) represents amount that indices are offset from being 0-indexed;
            e.g., if array_of_indices = [1, 3, 2, 5] and offset = 1, then those offset indices refer to
            sequence[0], sequence[2], sequence[1], sequence[4]
        mark = (Optional, default None) controls how bad indices present in array_of_indices are handled:
            if mark is None, they are deleted; otherwise they are set to mark
    Output:
        update_indices(array_of_indices, sorted_bad_indices, offset=offset) = new array of indices (with possible
            offset) referring to post-deletion sequence; bad indices present in array_of_indices are either
            deleted (if mark is None) or set to mark (otherwise)
    """
    if mark is None:
        delete = True
        mark = offset - 1
    else:
        delete = False

    # Store array_of_indices dtype and temporarily set it to int if it isn't already
    old_dtype = array_of_indices.dtype
    array_of_indices = array_of_indices.astype(int)

    # Instantiate new_indices to map old indices to
    new_indices = np.arange(offset, size + offset)

    # Pad sorted_bad_indices to create partition of new_indices
    padded_bad_indices = [0] + sorted_bad_indices + [size]

    # On each sub-interval of new_indices, decrement by number of bad_indices already encountered
    for index in range(len(padded_bad_indices) - 1):
        new_indices[
            padded_bad_indices[index]: padded_bad_indices[(index + 1)]
        ] -= index
        if index > 0:
            new_indices[padded_bad_indices[index]] = mark  # Mark bad indices
    updated_array = new_indices[
        array_of_indices - np.full(len(array_of_indices), offset)
    ]
    if delete:
        present_bad_indices = [
            index
            for index in range(len(updated_array))
            if updated_array[index] == offset - 1
        ]
        updated_array = np.delete(updated_array, present_bad_indices)

    # Return dtype to original value
    updated_array = updated_array.astype(old_dtype)

    return updated_array


add = operator.add
times = operator.mul
mul = operator.mul


def first(object_1: Any, _: Any) -> Any:
    """Binary projection onto first coordinate (Return first argument)."""
    return object_1


def last(_: Any, object_2: Any) -> Any:
    """Binary projection onto last coordinate (Return last argument)."""
    return object_2


# Aliases for valid binary operations
def operation_dict() -> Dict[str, Callable]:
    op_dict = {
        "add": add,
        "plus": add,
        "sum": add,
        "addition": add,
        "times": times,
        "mul": times,
        "multiply": times,
        "product": times,
        "multiplication": times,
        "prod": times,
        "min": min,
        "minimum": min,
        "minimize": min,
        "max": max,
        "maximum": max,
        "maximize": max,
        "first": first,
        "last": last,
    }
    return op_dict


def catstr(
    str_array_1: np.ndarray,
    str_array_2: np.ndarray,
    separator: str = "|",
    int_aware: bool = True,
) -> np.ndarray:
    """Concatenate arrays of strings/numbers str_array_1 and str_array_2 with separator sep between them."""
    str_array_1, str_array_2 = sanitize(str_array_1), sanitize(str_array_2)
    str_array_1 = num_to_str(str_array_1, int_aware=int_aware)
    str_array_2 = num_to_str(str_array_2, int_aware=int_aware)
    separator_array = np.full(len(str_array_1), separator)
    str_array_1_separator = np.char.add(str_array_1, separator_array)
    concatenation = np.char.add(str_array_1_separator, str_array_2)
    return concatenation


def get_default_args(func, *arg_names):
    default_args = dict()
    for arg_name in arg_names:
        default_args[arg_name] = signature(func).parameters[arg_name].default
    return default_args


def replace_default_args(func, **arg_pairs):
    new_args = list()
    default_args = get_default_args(func, *arg_pairs.keys())
    for arg_name in arg_pairs.keys():
        if arg_pairs[arg_name] is None:
            new_args.append(default_args[arg_name])
        else:
            new_args.append(arg_pairs[arg_name])
    if len(new_args) == 1:
        return new_args[0]
    else:
        return tuple(new_args)
