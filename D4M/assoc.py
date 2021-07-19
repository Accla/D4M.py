# Import packages
from __future__ import print_function, division

from scipy import sparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numbers
import random
import string
import csv
import shutil
from typing import Any, Union, Tuple, Optional


# Auxiliary/Helper functions

def string_gen(length_: int) -> str:
    """Create randomly-generated string of given length."""
    rand_string = ''
    for _ in range(length_):
        rand_string += random.SystemRandom().choice(string.ascii_letters)
    return rand_string


def num_string_gen(length_: int, upper_bound: int) -> str:
    """Create string list of integers <= upper_bound of given length."""
    rand_string = [str(random.randint(0, upper_bound)) for _ in range(length_)] + ['']
    rand_string = ','.join(rand_string)
    return rand_string


def is_numeric(object_: Any) -> bool:
    """ Check if object_ is numeric (int, float, complex, etc) or not. """
    if isinstance(object_, numbers.Number):
        numerical = True
    else:
        numerical = False
    return numerical


def sorted_union(array_1: np.ndarray, array_2: np.ndarray, return_index: Optional[bool] = None) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
    if return_index is None:
        return_index = False

    union = list()
    index_map_1 = list()
    index_map_2 = list()
    index_1 = 0
    index_2 = 0

    size_1 = np.size(array_1)
    size_2 = np.size(array_2)
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


def sorted_intersect(array_1: np.ndarray, array_2: np.ndarray, return_index: Optional[bool] = None,
                     return_index_1: Optional[bool] = None, return_index_2: Optional[bool] = None) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return the intersection of two sorted numpy arrays with index maps
    (if return_index, return_index_1, or return_index_2 are True).
        Usage:
            intersection = sorted_intersection(array_1, array_2)
        Input:
            array_1 = sorted numpy array of values with no duplicates
            array_2 = sorted numpy array of values with no duplicates
            return_index = boolean
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
    if return_index is None:
        return_index = False
    if return_index_1 is None:
        return_index_1 = False
    if return_index_2 is None:
        return_index_2 = False

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


def contains(substrings: Union[str, list[str]]) -> callable:
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
    substrings = sanitize(substrings)

    def func(string_list):
        string_list = sanitize(string_list)
        good_string_list = list()
        for index in range(len(string_list)):
            item = string_list[index]
            for substring in substrings:
                if substring in item:
                    good_string_list.append(index)
                    break
        return good_string_list

    return func


def startswith(prefixes: Union[str, list[str]]) -> callable:
    """Return callable which accepts a list of strings and returns the list of indices
    of those strings which have some element of prefixes as a prefix.
        Usage:
            startswith("a,b,")
            startswith(['a','b'])
        Inputs:
            prefixes = string of (delimiter separated) values (delimiter is last character)
                or list of values of length n
        Outputs:
            func(listofstrings) = returns a list of indices of the strings in listofstrings which have some element of
                prefixes as a prefix
    """
    prefixes = sanitize(prefixes)

    def func(string_list):
        string_list = sanitize(string_list)
        good_string_list = list()

        for index in range(len(string_list)):
            item = string_list[index]
            for prefix in prefixes:
                if item.startswith(prefix):
                    good_string_list.append(index)
                    break
        return good_string_list

    return func


def str_to_num(object_: str, delimiter: Optional[str] = None) -> Union[str, numbers.Number]:
    """Convert string to float if possible, otherwise return original object with optionally appended delimiter."""
    if isinstance(object_, str):
        try:
            object_ = int(object_)
        except ValueError:
            try:
                object_ = float(object_)
            except ValueError:
                if delimiter is not None:
                    object_ += delimiter
    return object_


def num_to_str(array: np.ndarray) -> np.ndarray:
    """Convert array of numbers to array of strings."""
    stringified_array = array.astype('str')
    return stringified_array


def sanitize(object_: Any, convert: Optional[bool] = None) -> np.ndarray:
    """Convert
        * strings of (delimiter-separated) values into a numpy array of values (delimiter = last character),
        * iterables into numpy arrays, and
        * all other objects into a numpy array having that object.
        Usage:
            sanitized list = sanitize(obj)
        Inputs:
            object_ = string of (delimiter separated) values (delimiter is last character)
                or iterable of values of length n or single value
            convert = Boolean indicating whether strings which represent numbers should
                be replaced with numbers
        Outputs:
            list of values
        Examples:
            sanitize("a,b,") = np.array(['a', 'b'])
            sanitize("1,1,") = np.array([1, 1])
            sanitize([10, 3]) = np.array([10, 3])
            sanitize(1) = np.array([1])
    """
    if convert is None:
        convert = False

    # Convert delimiter-separated string list by splitting using last character
    try:
        delimiter = object_[-1]
        object_ = object_.split(delimiter)
        object_.pop()  # Get rid of empty strings

        # Convert to numbers if requested
        if convert:
            object_ = [str_to_num(item) for item in object_]  # Convert applicable items to numbers
    except (AttributeError, IndexError, TypeError):
        pass

    # Convert to numpy array
    if not isinstance(object_, np.ndarray):
        if hasattr(object_, '__iter__'):
            object_ = np.array(object_)  # Possible silent upcasting
        else:
            object_ = np.array([object_])

    return object_


def length(iterable: Union[list, np.ndarray]) -> int:
    """Return length of iterable."""

    if isinstance(iterable, np.ndarray):
        return np.size(iterable)
    else:
        return len(iterable)


def unique(iterable: Union[list[Any], np.ndarray], return_index: Optional[bool] = None,
           return_inverse: Optional[bool] = None)\
        -> Union[list[Any], Tuple[list[Any], list[int]], Tuple[list[Any], list[int], list[int]],
                 np.ndarray, Tuple[np.ndarray, list[int]], Tuple[np.ndarray, list[int], list[int]]]:
    """Uniquiefy and sorts an iterable, optionally providing index maps."""
    if return_index is None:
        return_index = False
    if return_inverse is None:
        return_inverse = False

    if isinstance(iterable, np.ndarray):
        return np.unique(iterable, return_index=return_index, return_inverse=return_inverse)
    else:  # Assume iterable is a list

        # If no index maps needed, extract unique items in iterable and sort
        if not (return_index or return_inverse):
            return sorted(list(dict.fromkeys(iterable)))

        # If both index maps are needed, loop to extract unique items and build partial maps, then sort
        elif return_index and return_inverse:
            sorted_unique = list()
            seen = dict()
            index_map_unique = list()
            index_map_unique_inverse = list()
            latest = 0
            for index in range(len(iterable)):
                item = iterable[index]
                if item in seen.keys():
                    index_map_unique_inverse.append(seen[item])
                else:
                    index_map_unique_inverse.append(latest)
                    index_map_unique.append(index)
                    seen[item] = latest
                    latest += 1
                    sorted_unique.append(item)

            sorting_map = sorted(range(len(sorted_unique)), key=lambda k: sorted_unique[k])
            sorting_map_inverse = list(np.arange(len(sorting_map))[np.argsort(sorting_map)])
            sorted_unique = [sorted_unique[index] for index in sorting_map]

            index_map = [index_map_unique[index] for index in sorting_map]
            index_map_inverse = [sorting_map_inverse[index] for index in index_map_unique_inverse]

            return sorted_unique, index_map, index_map_inverse

        # Same as above but do not build index_map_inverse
        elif return_index and not return_inverse:
            sorted_unique = list()
            seen = set()
            index_map_unique = list()
            for index in range(len(iterable)):
                item = iterable[index]
                if item in seen:
                    pass
                else:
                    seen.add(item)
                    index_map_unique.append(index)
                    sorted_unique.append(item)

            sorting_map = sorted(range(len(sorted_unique)), key=lambda k: sorted_unique[k])
            sorted_unique = [sorted_unique[index] for index in sorting_map]

            index_map = [index_map_unique[index] for index in sorting_map]

            return sorted_unique, index_map

        # Same as above, but do not build index_map
        else:
            sorted_unique = list()
            seen = dict()
            index_map_unique_inverse = list()
            latest = 0
            for index in range(len(iterable)):
                item = iterable[index]
                if item in seen.keys():
                    index_map_unique_inverse.append(seen[item])
                else:
                    index_map_unique_inverse.append(latest)
                    seen[item] = latest
                    latest += 1
                    sorted_unique.append(item)

            sorting_map = sorted(range(len(sorted_unique)), key=lambda k: sorted_unique[k])
            sorting_map_inverse = list(np.arange(len(sorting_map))[np.argsort(sorting_map)])
            sorted_unique = [sorted_unique[index] for index in sorting_map]

            index_map_inverse = [sorting_map_inverse[index] for index in index_map_unique_inverse]

            return sorted_unique, index_map_inverse


def aggregate(row: Union[np.ndarray, list], col: Union[np.ndarray, list], val: Union[np.ndarray, list],
              func: callable) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate (row[i], col[i], val[i]) triples using func as collision function.
            Usage:
                aggregate(row, col, val, func)
            Inputs:
                row = numpy array of length n
                col = numpy array of length n
                val = numpy array of length n
                func = collision function (e.g. add, times, max, min, first, last)
            Output:
                newrow, newcol, newval = subarrays of row, col, val in which pairs (r, c) = (newrow[i], newcol[i])
                                            are unique and newval[i] is the resulting of iteratively
                                            applying func to the values corresponding to triples
                                            (r, c, value) = (row[j], col[j], val[j])
            Example:
                aggregate(['a', 'a', 'b'], ['A', 'A', 'B'], [1, 2, 3], add) = ['a', 'b'], ['A', 'B'], [3, 3]
                aggregate(['a', 'a', 'b'], ['A', 'A', 'B'], [1, 2, 3], first) = ['a', 'b'], ['A', 'B'], [1, 3]
                aggregate(['a', 'a', 'b'], ['A', 'A', 'B'], [1, 2, 3], last) = ['a', 'b'], ['A', 'B'], [2, 3]
                aggregate(['a', 'a', 'a', 'b'], ['A', 'A', 'A', 'B'], [1, 2, 0, 3], min)
                        = ['a', 'b'], ['A', 'B'], [0, 3]
                (where lists are stand-ins for the corresponding numpy arrays)
    """
    aggregate_dict = dict()
    for index in range(np.size(row)):
        if (row[index], col[index]) not in aggregate_dict:
            aggregate_dict[(row[index], col[index])] = val[index]
        else:
            aggregate_dict[(row[index], col[index])] = func(aggregate_dict[(row[index], col[index])], val[index])

    newrow = np.array([item[0] for item in list(aggregate_dict.keys())])
    newcol = np.array([item[1] for item in list(aggregate_dict.keys())])
    newval = np.array(list(aggregate_dict.values()))
    return newrow, newcol, newval


def add(object_1: Any, object_2: Any) -> Any:
    """Binary addition (including string concatenation or other implementations of __add__)."""
    return object_1 + object_2


def times(object_1: Any, object_2: Any) -> Any:
    """Binary multiplication (and other implementations of __mult__)."""
    return object_1 * object_2


def first(object_1: Any, _: Any) -> Any:
    """Binary projection onto first coordinate (Return first argument)."""
    return object_1


def last(_: Any, object_2: Any) -> Any:
    """Binary projection onto last coordinate (Return last argument)."""
    return object_2


# Aliases for valid binary operations
operation_dict = {'add': add, 'plus': add, 'sum': add, 'addition': add,
                  'times': times, 'multiply': times, 'product': times, 'multiplication': times,
                  'min': min, 'minimum': min, 'minimize': min, 'max': max, 'maximum': max, 'maximize': max,
                  'first': first, 'last': last}


def catstr(str_array_1: np.ndarray, str_array_2: np.ndarray, separator: Optional[str] = None) -> np.ndarray:
    """Concatenate arrays of strings/numbers str_array_1 and str_array_2 with separator sep between them."""
    if separator is None:
        separator = '|'

    str_array_1 = num_to_str(str_array_1)
    str_array_2 = num_to_str(str_array_2)
    separator_array = np.full(1, separator)
    str_array_1_separator = np.char.add(str_array_1, separator_array)
    concatenation = np.char.add(str_array_1_separator, str_array_2)
    return concatenation


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
    """

    null_values = {'', 0, None}

    def __init__(self, row: Union[str, list, np.ndarray], col: Union[str, list, np.ndarray],
                 val: Union[numbers.Number, str, list, np.ndarray],
                 arg: Optional[Union[sparse.spmatrix, callable, str]] = None):
        """Construct an associative array either from an existing sparse matrix (scipy.sparse.spmatrix) or
        from row, column, and value triples.
            Usage:
                A = Assoc(row,col,val)
                A = Assoc(row,col,val,func)
                A = Assoc(row,col,number,func)
                A = Assoc(row,col,val,sparse_matrix)
            Inputs:
                row = string of (delimiter separated) values (delimiter is last character)
                    or list of values of length n
                col = string of (delimiter separated) values (delimiter is last character)
                    or list of values of length n
                val = string of (delimiter separated) values (delimiter is last character)
                    or list of values of length n
                    or 1.0 (which signals arg to be a sparse matrix)
                    or other single value
                arg = either
                        a sparse matrix (to be used as the adjacency array) where
                            - if val=1.0, then arg is expected to contain the _actual_ values
                            - otherwise, val is expected to be a list of _actual_ values;
                                unique sorted entries in row, col, val are extracted
                                and the row/column indices and values of arg are assumed to match
                                up with the resulting row, col, val
                        or a two-input callable compatible with values,
                        or a string representing collision function, e.g., 'add', 'first', 'last', 'min', 'max'
                        or 'unique' which assumes there are no collisions
                    (default is min)
            Outputs:
                A = Associative array made from the triples row, col, and val
            Examples:
                A = Assoc('r1,r2,', 'c1,c2,', 'v1;v2;')
                A = Assoc(['r1','r2'], 'c1/c2/', [1,2], 'add')
                A = Assoc('1,', 'c1,', np.array([3]))
                A = Assoc('r1,r2,', 'c1,c2,', 1.0, [sparse_matrix])
            Notes:
                - If expected data is numerical, arg=add gives slight speed-up
                - If val == 1.0 and optional sparse matrix is supplied, it will be used as the adjacency array
                    where the row and column indices of the sparse matrix will be assumed to correspond
                    to the ordered row and col. Will throw an error if they aren't of the appropriate sizes.
                - To determine whether the data is numerical, values are sorted and the last element is examined.
                    From testing, Numpy appears to sort all non-numerical data types (strings, arrays, lists,
                    dicts, tuples, sets) to come after numerical data, so this should indicate whether there is
                    any non-numerical data.
        """
        if arg is None:
            arg = min
        elif sparse.issparse(arg) or callable(arg) or arg == 'unique':
            pass
        elif arg in operation_dict.keys():
            arg = operation_dict[arg]
        else:
            raise ValueError('Optional arg not supported.')

        # Sanitize
        row = sanitize(row)
        col = sanitize(col)

        row_size = np.size(row)
        col_size = np.size(col)

        # Short-circuit if empty assoc
        if row_size == 0 or col_size == 0 or np.size(val) == 0:
            self.row = np.empty(0)
            self.col = np.empty(0)
            self.val = 1.0  # Considered numerical
            self.adj = sparse.coo_matrix(([], ([], [])), shape=(0, 0))  # Empty sparse matrix
        else:
            # Handle data

            if sparse.issparse(arg):
                arg.eliminate_zeros()

                if isinstance(val, float) and val == 1.0:
                    is_float = True
                    arg.sum_duplicates()
                    val = arg.data
                else:
                    is_float = False
                    val = sanitize(val, convert=True)

                (row_dim, col_dim) = arg.shape

                unique_row = np.unique(row)
                unique_col = np.unique(col)
                unique_val = np.unique(val)

                error_message = 'Invalid input:'
                good_params = [np.size(unique_row) >= row_dim, np.size(unique_col) >= col_dim,
                               np.size(unique_val) >= np.size(np.unique(arg.data))]
                param_type = ['row indices', 'col indices', 'values']
                for index in range(3):
                    if index > 0 and False in good_params[0:index] and not good_params[index]:
                        error_message += ','
                    else:
                        pass
                    if good_params[index]:
                        pass
                    else:
                        error_message += ' not enough unique ' + param_type[index]
                error_message += '.'
                if False in good_params:
                    raise ValueError(error_message)
                else:
                    pass

                new_row = unique_row[arg.row]
                new_col = unique_col[arg.col]
                new_val = 0

                if is_float:
                    new_val = arg.data
                else:
                    try:
                        new_val = unique_val[arg.data - np.ones(np.size(arg.data), dtype=int)]
                    except (TypeError, IndexError):
                        print('Values in sparse matrix must correspond to elements of val (after sorting and removing '
                              'duplicates)')

                row = new_row
                row_size = np.size(row)
                col = new_col
                col_size = np.size(col)
                val = new_val
                arg = min
            else:
                pass

            val = sanitize(val)
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
                raise ValueError("Invalid input: row, col, val must have compatible lengths.")
            else:
                pass

            row, col, val = aggregate(row, col, val, arg)

            null_indices = [index for index in range(np.size(val)) if val[index] in Assoc.null_values]
            row = np.delete(row, null_indices)
            col = np.delete(col, null_indices)
            val = np.delete(val, null_indices)

            # Array possibly empty after deletion of null values
            if row_size == 0 or col_size == 0 or np.size(val) == 0:
                self.row = np.empty(0)
                self.col = np.empty(0)
                self.val = 1.0  # Considered numerical
                self.adj = sparse.coo_matrix(([], ([], [])), shape=(0, 0))  # Empty sparse matrix
            else:
                # Get unique sorted row and column indices
                self.row, from_row = np.unique(row, return_inverse=True)
                self.col, from_col = np.unique(col, return_inverse=True)
                self.val, from_val = np.unique(val, return_inverse=True)

                # Check if numerical; numpy sorts numerical values to front, so only check last entry
                assert isinstance(self.val, np.ndarray)
                if is_numeric(self.val[-1]):
                    self.adj = sparse.coo_matrix((val, (from_row, from_col)), dtype=float,
                                                 shape=(np.size(self.row), np.size(self.col)))
                    self.val = 1.0
                else:
                    # If not numerical, self.adj has entries given by indices+1 of self.val
                    val_indices = from_val + np.ones(np.size(from_val))
                    self.adj = sparse.coo_matrix((val_indices, (from_row, from_col)), dtype=int)

    def find(self, ordering: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get row, col, and val arrays that would generate the Assoc (reverse constructor).
                Usage:
                    self.find()
                Input:
                    self = Associative Array
                    ordering = optional parameter to control the ordering of result.
                                if 0, then order by row first, then column
                                if 1, then order by column first, then row
                                if None, no particular order is guaranteed
                Output:
                    row,col,val = numpy arrays for which self = Assoc(row, col, val)
        """
        # Use self.adj to extract row, col, and val
        if ordering == 0:  # Order by row first, then column
            row_adj = self.adj.tocsr().tocoo()
            enc_val, enc_row, enc_col = row_adj.data, row_adj.row, row_adj.col
        elif ordering == 1:  # Order by column first, then row
            col_adj = self.adj.tocsc().tocoo()
            enc_val, enc_row, enc_col = col_adj.data, col_adj.row, col_adj.col
        else:  # Otherwise don't order
            enc_val, enc_row, enc_col = self.adj.data, self.adj.row, self.adj.col

        if np.size(enc_row) != 0:
            row = self.row[enc_row]
        else:
            row = np.array([])
        if np.size(enc_col) != 0:
            col = self.col[enc_col]
        else:
            col = np.array([])

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

    def dict(self) -> dict:
        """Return two-dimensional dictionary adjacency_dict for which adjacency_dict[index1][index2]=value."""
        row, col, val = self.find()
        adjacency_dict = dict()

        for index in range(np.size(row)):
            if row[index] not in adjacency_dict:
                adjacency_dict[row[index]] = {col[index]: val[index]}
            else:
                adjacency_dict[row[index]][col[index]] = val[index]

        return adjacency_dict

    def getval(self) -> np.ndarray:
        """Return numpy array of unique values."""
        if isinstance(self.val, float):
            unique_values = np.unique(self.adj.data)
        else:
            unique_values = self.val

        return unique_values

    # print tabular form
    def printfull(self) -> None:
        """Print associative array in tabular form."""
        df = pd.DataFrame(self.dict())
        df = df.transpose()  # Transpose it to get correct order of rows/cols

        # Replace NaN's with empty string
        df.fillna('', inplace=True)

        print(df)
        return None

    def printfullalt(self) -> None:
        """Print associative array in tabular form (non-pandas implementation)."""
        terminal_col, terminal_row = shutil.get_terminal_size()

        # Determine if all rows fit in terminal window (with col labels and array size footer)
        cutoff = False
        if np.size(self.row) <= terminal_row - 3:
            max_array_rows = np.size(self.row)
        else:
            cutoff = True
            max_array_rows = terminal_row - 5  # Include space for vertical ellipses and final row

        rel_array = self[0:max_array_rows, :]  # Disregard rows outside range
        trans_dict = rel_array.transpose().dict()  # Take transpose to collate by column labels
        trans_dict.update(self[-1, :].transpose().dict())  # Add in final row (may already be present)

        # Find widths of columns (w.r.t. only pre-cutoff rows and last row, unless column is empty as a result)
        col_widths = list()
        for index in np.arange(np.size(self.col)):
            col_label = self.col[index]
            if col_label in trans_dict:
                width = max([len(str(val)) for val in trans_dict[col_label].values()])
            else:
                width = 0
            width = max([width, len(str(col_label))])
            col_widths.append(width)

        # Format array values according to calculated column-wise max widths
        rel_dict = rel_array.dict()
        rel_dict.update(self[-1, :].dict())  # Add in final row (may already be present)
        formatted_rows = list()
        row_indices = list(range(max_array_rows))
        if cutoff:
            row_indices.append(-1)
        else:
            pass
        for row_index in row_indices:
            current_row = list()
            for col_index in np.arange(np.size(self.col)):
                if self.col[col_index] in rel_dict[self.row[row_index]]:
                    row_item = str(rel_dict[self.row[row_index]][self.col[col_index]])
                else:
                    row_item = ''
                current_row.append(('  {:>' + str(col_widths[col_index]) + '}').format(row_item))
            formatted_rows.append(current_row)
        if cutoff:
            vellipses = list()
            for col_index in np.arange(np.size(self.col)):
                vellipses.append(('  {:>' + str(col_widths[col_index]) + '}').format(':'))  # ⋮
            formatted_rows.insert(-1, vellipses)
        else:
            pass

        # Format column labels according to calculated column-wise max widths
        col_labels = list()
        for index in np.arange(np.size(self.col)):
            col_labels.append(('  {:>' + str(col_widths[index]) + '}').format(str(self.col[index])))

        # Format row labels according to max (relevant) row label width
        row_label_width = max([len(str(self.row[row_index])) for row_index in row_indices])
        row_labels = list()
        for row_index in row_indices:
            row_labels.append(('{:<' + str(row_label_width) + '}').format(str(self.row[row_index])))
        if cutoff:
            row_labels.insert(-1, ('{:<' + str(row_label_width) + '}').format(':'))
        else:
            pass

        # Determine how many columns fit in terminal window and print
        last_index = np.size(self.col)-1
        too_wide = False

        # Case 1: All columns fits
        if row_label_width + sum(col_widths) + 2 * (last_index + 1) <= terminal_col:
            print(' '*row_label_width + ''.join(col_labels))
            for row_index in np.arange(len(row_labels)):
                print(row_labels[row_index] + ''.join(formatted_rows[row_index]))

        # Case 2: Not all columns fit, but at least the first and last do, plus ellipsis
        elif row_label_width + col_widths[0] + col_widths[last_index] + 9 <= terminal_col:

            # Determine how many columns fit
            penul_index = 0
            running_total = row_label_width + col_widths[last_index] + 9
            while running_total + col_widths[penul_index] + 2 <= terminal_col:
                running_total += col_widths[penul_index] + 2
                penul_index += 1

            print(' ' * row_label_width + ''.join(col_labels[0:penul_index]) + '  ...' + col_labels[last_index])
            for row_index in np.arange(len(row_labels)):
                print(row_labels[row_index] + ''.join(formatted_rows[row_index][0:penul_index])
                      + '  ...' + formatted_rows[row_index][last_index])

            too_wide = True

        # Case 3: First column plus ellipses fit, but not with addition of final column
        elif row_label_width + col_widths[0] + 7 <= terminal_col:

            # Determine how many columns fit
            penul_index = 0
            running_total = row_label_width + 7
            while running_total + col_widths[penul_index] + 2 <= terminal_col:
                running_total += col_widths[penul_index] + 2
                penul_index += 1

            print(' ' * row_label_width + ''.join(col_labels[0:penul_index]) + '  ...')
            for row_index in np.arange(len(row_labels)):
                print(row_labels[row_index] + ''.join(formatted_rows[row_index][0:penul_index]) + '  ...')

            too_wide = True

        # Case 4: Can't fit even the first column plus ellipsis
        else:
            print('Columns too wide to fit window.')

            too_wide = True

        # Report dimensions if either horizontally or vertically cut off
        if cutoff or too_wide:
            print('\n' + '[' + str(np.size(self.row)) + ' rows x ' + str(np.size(self.col)) + ' columns]')

        return None

    def spy(self) -> None:
        """Print spy plot of self.adj"""
        plt.spy(self.adj, markersize=0.2, aspect='auto')
        plt.show()
        return None

    # Overload print
    def __str__(self) -> str:
        """Print the attributes of associative array."""
        print_string = "Row indices: " + str(self.row) + "\n"
        print_string += "Column indices: " + str(self.col) + "\n"
        print_string += "Values: " + str(self.val) + "\n"
        print_string += "Adjacency array: " + "\n" + str(self.adj.toarray())
        return print_string

    def triples(self, ordering: Optional[int] = None) -> list:
        """Return list of triples of form (row_label,col_label,value)."""
        r, c, v = self.find(ordering=ordering)
        triples = list(zip(list(r), list(c), list(v)))
        return triples

    def getvalue(self, row_key: Any, col_key: Any) -> Any:
        """Get the value in self corresponding to given row_key and col_key, otherwise return 0.
            Usage:
                v = A.getvalue('a', 'B')
            Inputs:
                A = self = Associative Array
                row_key = row key
                col_key = column key
            Output:
                v = value of A corresponding to the pair (row_key, col_key),
                    i.e., (row_key, col_key, v) is in A.triples()
            Note:
                If either of row_key or col_key are integers, they are taken as indices instead of *actual*
                row and column keys, respectively.
        """
        if not isinstance(row_key, int) and row_key in self.row:
            index_1 = np.where(self.row == row_key)[0][0]
        else:
            index_1 = row_key

        if not isinstance(col_key, int) and col_key in self.col:
            index_2 = np.where(self.col == col_key)[0][0]
        else:
            index_2 = col_key

        if isinstance(self.val, float):
            try:
                return self.adj.todok()[index_1, index_2]
            except (IndexError, TypeError):
                return 0
        else:
            assert isinstance(self.val, np.ndarray)
            try:
                return self.val[self.adj.todok()[index_1, index_2] - 1]
            except (IndexError, TypeError):
                return 0

    # Overload getitem; allows for subsref
    def __getitem__(self, selection) -> 'Assoc':
        """Returns a sub-associative array of self according to object1 and object2 or corresponding value
            Usage:
                B = A[row_select, col_select]
            Inputs:
                A = Associative Array
                obj = tuple (row_select, col_select) where
                    row_select = string of (delimiter separate) values (delimiter is last character)
                        or iterable or int or slice object or function
                    col_select = string of (delimiter separate) values (delimiter is last character)
                        or iterable or int or slice object or function
                        e.g., "a,:,b,", "a,b,c,d,", ['a',':','b'], 3, [1,2], 1:2, startswith("a,b,"),
                            "a *,"
            Outputs:
                B = sub-associative array of A whose row indices are selected by row_select and whose
                    column indices are selected by col_select, assuming not both of row_select, col_select are single
                    indices
                B = value of A corresponding to single indices of row_select and col_select
            Examples:
                A['a,:,b,', ['0', '1']]
                A[1:2:1, 1]
            Note:
                - Regular slices are NOT right end-point inclusive
                - 'Slices' of the form "a,:,b," ARE right end-point inclusive (i.e. includes b)
                - Integer row_select or col_select, and by extension slices, do not reference A.row or A.col,
                    but the induced indexing of the rows and columns
                    e.g., A[:, 0:2] will give the subarray consisting of all rows and the columns col[0], col[1],
                        A[:, 0] will give the subarray consisting of the 0-th column
                        A[2, 4] will give the value in the 2-nd row and 4-th column
        """
        keys = [self.row, self.col]
        row_select, col_select = selection
        selection = [row_select, col_select]

        # For each object, replace with corresponding array of row/col keys
        for index in [0, 1]:
            i_select = selection[index]
            # If object is a single integer, replace with corresponding row/col key
            if isinstance(i_select, int):
                selection[index] = [keys[index][i_select]]
                continue

            # If object is an iterable of integers, replace with corresponding row/col keys
            all_integers = True
            if hasattr(i_select, '__iter__'):
                for item in i_select:
                    if not isinstance(item, int):
                        all_integers = False
                        break
            else:
                all_integers = False

            if all_integers:
                selection[index] = keys[index][i_select]
                continue
            else:
                pass

            # If object is a function on iterables returing list of indices, apply it
            if callable(i_select):
                selection[index] = keys[index][i_select(keys[index])]
                continue
            else:
                pass

            # If object is a slice object, convert to appropriate list of keys
            if isinstance(i_select, slice):
                selection[index] = keys[index][i_select]
                continue
            else:
                pass

            # If object is of form ":", convert to appropriate list of keys
            if isinstance(i_select, str):
                if i_select == ":":
                    selection[index] = keys[index]
                    continue
            else:
                pass

            # Then, or otherwise, sanitize to get appropriate list of keys
            i_select = sanitize(i_select)

            # If resulting object is 'slice-like', replace with appropriate list of keys,
            # getting all keys where i_select[0] <= element <= i_select[2]
            # so find first index of key with i_select[0] <= key and first index of key with
            # i_select[2] < key (so all earlier keys are <= i_select[2]).

            if len(i_select) == 3 and i_select[1] == ":":
                start_compare = (keys[index] >= i_select[0])
                stop_compare = (keys[index] > i_select[2])
                try:
                    start_index = np.argwhere(start_compare)[0][0]
                except IndexError:
                    start_index = np.size(keys[index])
                try:
                    stop_index = np.argwhere(stop_compare)[0][0]
                except IndexError:
                    stop_index = np.size(keys[index])
                selection[index] = keys[index][start_index:stop_index]
            else:
                pass

            selection[index] = i_select

        row_select, col_select = selection  # Now everything is a list of row/col keys

        # Create new row, col, val triple to construct sub-assoc array
        row_select = np.sort(row_select)
        col_select = np.sort(col_select)

        new_row, row_index_map = sorted_intersect(self.row, row_select, return_index_1=True)
        new_col, col_index_map = sorted_intersect(self.col, col_select, return_index_1=True)

        subarray = Assoc([], [], [])
        subarray.row = np.array(new_row)
        subarray.col = np.array(new_col)
        subarray.val = self.val
        subarray.adj = self.adj.tocsr()[row_index_map, :][:, col_index_map].tocoo()

        subarray = subarray.condense()
        subarray = subarray.deepcondense()

        return subarray

    # Overload setitem
    def __setitem__(self, col_index, row_index, value):
        return NotImplemented

    def copy(self) -> 'Assoc':
        """Create a copy of self."""
        array_copy = Assoc([], [], [])
        array_copy.row = self.row.copy()
        array_copy.col = self.col.copy()
        if isinstance(self.val, float):
            array_copy.val = 1.0
        else:
            assert isinstance(self.val, np.ndarray)
            array_copy.val = self.val.copy()
        array_copy.adj = self.adj.copy()

        return array_copy

    def size(self) -> Tuple[int, int]:
        """Returns dimensions of self."""
        size1 = np.size(self.row)
        size2 = np.size(self.col)
        return size1, size2

    def nnz(self):
        """
        Returns number of stored entries.
            Usage:
                A.nnz()
            Input:
                self = Associative array
            Output:
                The number of stored entries (i.e., the number of entries in the adjacency array).
            Notes:
                - If null values (0, None, '') are explicitly stored in the array, then they are counted. To count only
                    non-null values, use self.get_nonzero()
        """
        nnz = self.adj.getnnz()
        return nnz

    def count_nonzero(self):
        """Returns number of non-null stored entries.
            Usage:
                A.count_nonzero()
            Input:
                self = Associative array
            Output:
                The number of stored non-null entries (i.e., the number of non-null entries in the adjacency array).
            Notes:
                - Even if null values (0, None, '') are explicitly stored in the array, they are not counted.
        """
        null = [0, None, '']

        # If data is numerical, return number of nonzero values.
        if isinstance(self.val, float):
            nonzero = np.where(self.adj.data)
            actual_nnz = np.size(nonzero)
        # Otherwise, check against list of null values
        else:
            _, _, val = self.find()
            nonzero = [value not in null for value in val]
            actual_nnz = np.size(val[nonzero])

        return actual_nnz

    # Remove zeros/empty strings/None from being recorded
    def dropzeros(self, copy=False):
        """
        Return copy of Assoc without null values recorded.
            Usage:
                A.dropzeros()
                A.dropzeros(copy = True)
            Inputs:
                self = Associative array
                copy = Whether a new Assoc instance should be made or
                    if existing instance should be modified
            Outputs:
                Associative subarray of A consisting only of non-null values
            Notes:
                - Null values include 0, '', and None
        """

        # Enumerate the 'null' values
        null = ['', 0, None]

        # If numerical, just use scipy.sparse's eliminate_zeros()
        if isinstance(self.val, float):
            if not copy:
                A = self
            else:
                A = self.copy()

            # Remove zeros and update row and col appropriately
            A.adj.eliminate_zeros()
            A.condense()
        # Otherwise, manually remove and remake Assoc instance
        else:
            if not copy:
                A = self
            else:
                A = Assoc([], [], [])

            row, col, val = self.find()

            # Determine which values are non-zero
            good_indices = [value not in null for value in val]

            # Remove the row/col/val triples that correspond to a zero value
            row = row[good_indices]
            col = col[good_indices]
            val = val[good_indices]

            # Get unique sorted row and column indices
            A.row, fromrow = np.unique(row, return_inverse=True)
            A.col, fromcol = np.unique(col, return_inverse=True)
            A.val, fromval = np.unique(val, return_inverse=True)

            # Fix empty results
            if np.size(A.row) == 0:
                A.row = np.array([])
            if np.size(A.col) == 0:
                A.col = np.array([])
            if np.size(A.val) == 0:
                A.val = 1.0

            # Make adjacency array
            val_indices = fromval + np.ones(np.size(fromval))
            A.adj = sparse.coo_matrix((val_indices, (fromrow, fromcol)), dtype=int,
                                      shape=(np.size(A.row), np.size(A.col)))

        return A

    # Redefine adjacency array
    def setadj(self, new_adj):
        """
            Replace the adjacency array of self with new_adj. (in-place)
                Usage:
                    A.setadj(new_adj)
                Input:
                    self = Associative array
                    new_adj = A sparse matrix whose dimensions are at least that of self.
                Output:
                    self = Associative array with given sparse matrix as adjacency array
                        and row and column values cut down to fit the dimensions of the
                        new adjacency array
        """
        self.val = 1.0

        # Get shape of new_adj and cut down self.row/self.col to size
        (row_size, col_size) = new_adj.shape

        if np.size(self.row) < row_size or np.size(self.col) < col_size:
            raise ValueError("new_adj is too large for existing row and column indices.")
        else:
            self.row = self.row[0:row_size]
            self.col = self.col[0:col_size]
            self.adj = new_adj.tocoo()

        return self

    # Get diagonal; output as a numpy array
    def diag(self):
        """ Output the diagonal of self.arg as a numpy array. """
        enc_diag = self.adj.diagonal()
        if isinstance(self.val, float):
            diag = enc_diag
        else:
            # Append 0 to the start of self.val so that indices of enc_kdiag match up
            inc_val = np.append(np.zeros(1), self.val)
            diag = inc_val[enc_diag]
        return diag

    def sum(self, axis=None):
        """
            Sum over the given axis or over whole array if None.
                Usage:
                    A.sum()
                    A.sum(0)
                    A.sum(1)
                Input:
                    self = Associative array
                    axis = 0 if summing down columns, 1 if summing across rows, and None if over whole array
                Output:
                    A.sum(axis) = Associative array resulting from summing over indicated axis
        """

        # If any of the values are strings, convert to logical
        # In this case, the adjacency array is the desired sparse matrix to sum over
        if not isinstance(self.val, float):
            new_sparse = self.logical(copy=True).adj.copy()
        # Otherwise, build a new sparse matrix with actual (numerical) values
        else:
            new_sparse = self.adj.copy()

        # Sum as a sparse matrix over desired axis
        summed_sparse = new_sparse.sum(axis)

        # Depending on axis, build associative array
        if axis is None:
            A = summed_sparse
        elif axis == 1:
            A = Assoc([], [], [])
            A.row = self.row.copy()
            A.col = np.array([0])
            A.val = 1.0
            A.adj = sparse.coo_matrix(summed_sparse)

        elif axis == 0:
            A = Assoc([], [], [])
            A.row = np.array([0])
            A.col = self.col.copy()
            A.val = 1.0
            A.adj = sparse.coo_matrix(summed_sparse)
        else:
            A = None

        return A

    # replace all non-zero values with ones
    def logical(self, copy=True):
        """
            Replaces every non-zero value with 1.0
                Usage:
                    A.logical()
                    A.logical(copy=False)
                Input:
                    self = Associative array
                    copy = boolean indicating whether the operation is in-place or not
                Output:
                    self.logical() = a copy of self with all non-zero values replaced with 1.0
                    self.logical(copy=False) = self with all non-zero values replaced with 1.0
        """
        if copy:
            A = self.dropzeros(copy=True)
        else:
            A = self.dropzeros()

        A.val = 1.0

        A.adj.data[:] = 1.0
        return A

    # Overload element-wise addition
    def __add__(self, B):
        """
            Element-wise addition of self and B, matched up by row and column indices.
                Usage:
                    A + B
                Input:
                    A = self = Associative array
                    B = Associative array
                Output:
                    A + B =
                        * element-wise sum of A and B (if both A and B are numerical)
                        * element-wise minimum of A and B otherwise
                Note:
                    - When either A or B are non-numerical the .logical() method is run on them.
        """

        A = self

        if isinstance(A.val, float) and isinstance(B.val, float):
            # Take union of rows and cols while keeping track of indices
            row_union, row_index_A, row_index_B = sorted_union(A.row, B.row, return_index=True)
            col_union, col_index_A, col_index_B = sorted_union(A.col, B.col, return_index=True)

            row = np.append(row_index_A[A.adj.row], row_index_B[B.adj.row])
            col = np.append(col_index_A[A.adj.col], col_index_B[B.adj.col])
            val = np.append(A.adj.data, B.adj.data)

            # Make sparse matrix and sum duplicates
            C = Assoc([], [], [])
            C.row = row_union
            C.col = col_union
            C.val = 1.0
            C.adj = sparse.coo_matrix((val, (row, col)),
                                      shape=(np.size(row_union), np.size(col_union)))
            C.adj.sum_duplicates()
        else:
            # Take union of rows, cols, and vals
            rowA, colA, valA = A.find()
            rowB, colB, valB = B.find()
            row = np.append(rowA, rowB)
            col = np.append(colA, colB)
            val = np.append(valA, valB)

            # Construct with min as collision function
            C = Assoc(row, col, val, 'min')

        return C

    def __sub__(self, B):
        """ Subtract array B from array A=self, i.e. A-B. """

        A = self
        C = B.copy()

        # If not numerical, convert to logical
        if not isinstance(A.val, float):
            A = A.logical(copy=True)
        if not isinstance(B.val, float):
            C = C.logical(copy=True)

        # Negate second array argument's numerical data
        C.adj.data = -C.adj.data

        D = A + C
        return D

    # Overload matrix multiplication
    def __mul__(self, B):
        """
            Array multiplication of A and B, with A's column indices matched up with B's row indices
                Usage:
                    A * B
                Input:
                    A = self = Associative array
                    B = Associative array
                Output:
                    A * B = array multiplication of A and B
                Note:
                    - When either A or B are non-numerical the .logical() method is run on them.
        """

        A = self
        # If either A=self or B are not numerical, replace with logical()
        if not isinstance(A.val, float):
            A = A.logical()
        if not isinstance(B.val, float):
            B = B.logical()

        # Convert to CSR format for better performance
        A_sparse = A.adj.tocsr()
        B_sparse = B.adj.tocsr()

        # Intersect A.col and B.row
        intersection, index_map_1, index_map_2 = sorted_intersect(A.col, B.row,
                                                                  return_index=True)

        # Get appropriate sub-matrices
        A_sparse = A_sparse[:, index_map_1]
        B_sparse = B_sparse[index_map_2, :]

        # Multiply sparse matrices
        AB_sparse = A_sparse * B_sparse

        # Construct Assoc array
        AB = Assoc([], [], [])  # Construct empty array
        AB.row = A.row
        AB.col = B.col
        AB.val = 1.0
        AB.adj = AB_sparse.tocoo()

        return AB

    # element-wise multiplication
    def multiply(self, B):
        """
            Element-wise multiplication of self and B, matched up by row and column indices.
                Usage:
                    A.multiply(B)
                Input:
                    A = self = Associative array
                    B = Associative array
                Output:
                    A + B = element-wise product of A and B
                Note:
                    - When either A or B are non-numerical the .logical() method is run on them.
        """

        A = self
        # Only multiply if both numerical, so logical() as appropriate
        if not isinstance(B.val, float):
            B = B.logical()
        if not isinstance(A.val, float):
            A = A.logical()

        row_int, row_index_A, row_index_B = sorted_intersect(A.row, B.row, return_index=True)
        col_int, col_index_A, col_index_B = sorted_intersect(A.col, B.col, return_index=True)

        C = Assoc([], [], [])
        C.row = row_int
        C.col = col_int
        C.val = 1.0
        Asub = A.adj.tocsr()[:, col_index_A][row_index_A, :]
        Bsub = B.adj.tocsr()[:, col_index_B][row_index_B, :]
        C.adj = Asub.multiply(Bsub).tocoo()

        return C

    def transpose(self, copy=True):
        """ Transpose array, switching self.row and self.col and transposing self.adj. """
        if copy:
            A = Assoc([], [], [])
            A.row = self.col.copy()
            A.col = self.row.copy()
            if isinstance(self.val, float):
                A.val = 1.0
            else:
                assert isinstance(self.val, np.ndarray)
                A.val = self.val.copy()
            A.adj = self.adj.copy().transpose()
        else:
            A = self
            temp = self.row.copy()
            self.row = self.col
            self.col = temp
            self.adj = self.adj.transpose()

        return A

    # Remove row/col indices that do not appear in the data
    def condense(self):
        """Remove items from self.row and self.col which do not correspond to values, according to self.adj.
                Usage:
                    A.condense()
                    B = A.condense()
                Input:
                    self = Associative array
                Output:
                    self = self.condense() = Associative array which removes all elements of self.row and self.col
                            which are not associated with some (nonzero) value.
                Notes:
                    - In-place operation.
                    - Elements of self.row or self.col which correspond to rows or columns of all 0's
                        (but not '' or None) are removed.
        """
        row, col, _ = self.find()

        # First do row, determine which indices in self.row show up in row, get index map, and select
        present_row = np.isin(self.row, row)
        index_map = np.where(present_row)[0]

        self.row = self.row[present_row]
        self.adj = self.adj.tocsr()[index_map, :].tocoo()  # Removes indices corresponding to zero rows

        # Col
        present_col = np.isin(self.col, col)
        index_map = np.where(present_col)[0]

        self.col = self.col[present_col]
        self.adj = self.adj.tocsr()[:, index_map].tocoo()  # Removes indices corresponding to zero cols

        return self

    # extension of condense() which also removes unused values
    def deepcondense(self):
        """Remove values from self.val which are not reflected in self.adj."""

        # If numerical, do nothing (no unused values)
        if isinstance(self.val, float):
            return self
        else:
            # Otherwise, re-run corresponding part of constructor

            # Get actually-used row,col,val
            row, col, val = self.find()

            # Get unique sorted row, column indices and values
            self.row, fromrow = np.unique(row, return_inverse=True)
            self.col, fromcol = np.unique(col, return_inverse=True)
            self.val, fromval = np.unique(val, return_inverse=True)

            # Remake adjacency array
            val_indices = fromval + np.ones(np.size(fromval))
            self.adj = sparse.coo_matrix((val_indices, (fromrow, fromcol)), dtype=int,
                                         shape=(np.size(self.row), np.size(self.col)))

            # If self.val is now empty, replace with 1.0
            if np.size(self.val) == 0:
                self.val = 1.0

            return self

    # Eliminate columns
    def nocol(self, copy=True):
        """ Eliminate columns.
            Usage:
                A.nocol()
                A.nocol(copy=False)
            Input:
                copy = boolean indicating whether operation should be in-place
            Output:
                A.nocol() = Associative array with same row indices as A and single column index 0.
                            The i-th row of A.nocol() is 1 only when the i-th row of A had a non-zero entry.
                A.nocol(copy=False) = in-place version
        """
        if copy:
            A = self.copy()
        else:
            A = self

        # Take logical, sum over rows, then logical again
        A.logical(copy=False)
        A = A.sum(1)
        A.logical(copy=False)
        A.col = np.array([0])

        return A

    # Eliminate rows
    def norow(self, copy=True):
        """ Eliminate rows.
            Usage:
                A.norow()
                A.norow(copy=False)
            Input:
                copy = boolean indicating whether operation should be in-place
            Output:
                A.norow() = Associative array with same col indices as A and single row index 0.
                            The i-th col of A.norow() is 1 only when the i-th col of A had a non-zero entry.
                A.norow(copy=False) = in-place version
        """
        if copy:
            A = self.copy()
        else:
            A = self

        # Take logical, sum over cols, then logical again
        A.logical(copy=False)
        A = A.sum(0)
        A.logical(copy=False)
        A.row = np.array([0])

        return A

    # element-wise division -- for division by zero, replace with 0 (and remove)
    def divide(self, B):
        """
            Element-wise division of self and B, matched up by row and column indices.
                Usage:
                    A.divide(B)
                Input:
                    A = self = Associative array
                    B = Associative array
                Output:
                    A.divide(B) = element-wise quotient of A by B
                Note:
                    - Removes all explicit zeros and ignores division by zero
                    - Implicitly runs .logical() method on non-numerical arrays
        """

        Binv = B.dropzeros(copy=True)
        Binv.adj.data = np.reciprocal(Binv.adj.data.astype(float, copy=False))

        C = self.multiply(Binv)

        return C

    # element-wise And
    def __and__(self, B):
        """
            Element-wise logical AND of self and B, matched up by row and column indices.
                Usage:
                    A & B
                Input:
                    A = self = Associative array
                    B = Associative array
                Output:
                    A & B = element-wise logical AND of A.logical() and B.logical()
        """

        A = self.logical(copy=True)
        B = B.logical(copy=True)

        C = A.multiply(B)
        return C

    # element-wise or
    def __or__(self, B):
        """
            Element-wise logical OR of self and B, matched up by row and column indices.
                Usage:
                    A | B
                Input:
                    A = self = Associative array
                    B = Associative array
                Output:
                    A | B = element-wise logical OR of A.logical() and B.logical()
        """

        A = self.logical(copy=True)
        B = B.logical(copy=True)

        C = A + B
        C = C.logical(copy=False)
        return C

    def sqin(self):
        """ self.transpose() * self """
        return self.transpose() * self

    def sqout(self):
        """ self * self.transpose() """
        return self * self.transpose()

    # CatKeyMul
    def catkeymul(self, B, delimiter=None):
        """
            Computes the array product, but values are delimiter-separated string list of
                the row/column indices which contribute to the value in the product
                Usage:
                    A.catkeymul(B)
                    A.catkeymul(B,delimiter)
                Input:
                    A = Associative array
                    B = Associative Array
                    delimiter = optional delimiter to separate the row/column indices. Default is semi-colon ';'
                Output:
                    A.catkeymul(B) = Associative array where the (i,j)-th entry is null unless the (i,j)-th entry
                        of A.logical() * B.logical()  is not null, in which case that entry is the string list of
                        the k-indices for which A[i,k] and B[k,j] were non-zero.
        """
        A = self
        intersection = sorted_intersect(A.col, B.row)

        Alog = A[:, intersection].logical()
        Blog = B[intersection, :].logical()
        C = Alog * Blog

        intersection = np.array([str(item) for item in intersection])

        row, col, val = C.find()
        catval = np.zeros(np.size(row), dtype=object)

        if delimiter is None:
            delimiter = ';'

        # Create dictionaries for faster lookups
        row_ind = {C.row[index]: index for index in range(np.size(C.row))}
        col_ind = {C.col[index]: index for index in range(np.size(C.col))}

        rows = Alog.adj.tolil().rows
        cols = Blog.adj.transpose().tolil().rows

        # Enumerate all the row/col key lists to be intersected
        row_keys = {r: intersection[rows[row_ind[r]]] for r in C.row}
        col_keys = {c: set(intersection[cols[col_ind[c]]]) for c in C.col}  # Use set for O(1) lookup

        # Instantiate dictionary to hold already-calculated intersections
        cat_inters = dict()

        for i in range(np.size(row)):
            r = row[i]
            c = col[i]
            if (r, c) in cat_inters:
                catval[i] = cat_inters[(r, c)]
            else:
                catval[i] = delimiter.join([item for item in row_keys[r]
                                            if item in col_keys[c]]) + delimiter
                cat_inters[(r, c)] = catval[i]

        D = Assoc(row, col, catval, add)

        return D

    def catvalmul(self, B, pair_delimiter=None, delimiter=None):
        """
            Computes the array product, but values are delimiter-separated string list of
                the values of A and B which contribute to the value in the product
                Usage:
                    A.catvalmul(B)
                    A.catvalmul(B,pair_delimiter=given1,delimiter=given2)
                Input:
                    A = Associative array
                    B = Associative Array
                    pair_delimiter = optional delimiter to separate the values in A and B. Default is comma ','
                    delimiter = optional delimiter to separate the value pairs. Default is semi-colon ';'
                Output:
                    A.catvalmul(B) = Associative array where the (i,j)-th entry is null unless the (i,j)-th entry
                        of A.logical() * B.logical()  is not null, in which case that entry is the string list of
                        the non-trivial value pairs 'A[i,k],B[k,j],'.
        """
        A = self

        intersection = sorted_intersect(A.col, B.row)

        Alog = A[:, intersection].logical()
        Blog = B[intersection, :].logical()
        C = Alog * Blog

        row, col, val = C.find()
        catval = np.zeros(np.size(row), dtype=object)

        if delimiter is None:
            delimiter = ';'
        if pair_delimiter is None:
            pair_delimiter = ','

        # Create dictionaries for faster lookups
        row_ind = {C.row[index]: index for index in range(np.size(C.row))}
        col_ind = {C.col[index]: index for index in range(np.size(C.col))}
        A_dict = A.dict()
        B_dict = B.dict()

        rows = Alog.adj.tolil().rows
        cols = Blog.adj.transpose().tolil().rows

        # Enumerate all the row/col key lists to be intersected
        row_keys = {r: intersection[rows[row_ind[r]]] for r in C.row}
        col_keys = {c: set(intersection[cols[col_ind[c]]]) for c in C.col}  # Use set for O(1) lookup

        # Instantiate dictionary to hold already-calculated intersections
        cat_inters = dict()

        for i in range(np.size(row)):
            r = row[i]
            c = col[i]
            if (r, c) in cat_inters:
                catval[i] = cat_inters[(r, c)]
            else:
                rc_keys = [item for item in row_keys[r] if item in col_keys[c]]
                rc_valpairs = [str(A_dict[r][key])
                               + pair_delimiter
                               + str(B_dict[key][c])
                               + pair_delimiter for key in rc_keys]
                catval[i] = delimiter.join(rc_valpairs) + delimiter
                cat_inters[(r, c)] = catval[i]

        D = Assoc(row, col, catval, add)

        return D

    def __eq__(self, other):
        """
            Element-wise equality comparison between self and other.
                Usage:
                    A == B
                Input:
                    A = Associative Array
                    B = other object, e.g., another associative array, a number, or a string
                Output:
                    A == B = An associative array such that for row and column labels r and c, resp., such that
                            (A == B)(r,c) = 1 if and only if...
                                (Case 1) A(r,c) == B(r,c) (when B is another associative array
                                    and assuming A(r,c) and B(r,c) are not null)
                                (Case 2) A(r,c) == B (when B is not another associative array)
                            otherwise (A == B)(r,c) = null.
                Notes:
                    Returns an error if other is null (0, '', or None)
        """
        null = [0, '', None]

        selfdict = self.dict()
        selfkeys = self.dict().keys()
        goodrows = list()
        goodcols = list()

        if isinstance(other, Assoc):
            otherdict = other.dict()
            otherkeys = otherdict.keys()

            for key in selfkeys:
                if key in otherkeys:
                    if selfdict[key] not in null and otherdict[key] not in null and selfdict[key] == otherdict[key]:
                        goodrows.append(key[0])
                        goodcols.append(key[1])
        elif other not in null:
            for key in selfkeys:
                if selfdict[key] == other:
                    goodrows.append(key[0])
                    goodcols.append(key[1])
        else:
            raise ValueError("Comparison with a null value")

        A = Assoc(goodrows, goodcols, 1)
        return A

    def __ne__(self, other):
        """
                    Element-wise inequality comparison between self and other.
                        Usage:
                            A != B
                        Input:
                            A = Associative Array
                            B = other object, e.g., another associative array, a number, or a string
                        Output:
                            A != B = An associative array such that for row and column labels r and c, resp., such that
                                    (A != B)(r,c) = 1 if and only if...
                                        (Case 1) A(r,c) != B(r,c) (when B is another associative array
                                                and at least one of A(r,c) and B(r,c) are not null)
                                        (Case 2) A(r,c) != B (when B is not another associative array)
                                    otherwise (A != B)(r,c) = null.
        """
        null = [0, '', None]

        selfdict = self.dict()
        selfkeys = self.dict().keys()
        goodrows = list()
        goodcols = list()

        if isinstance(other, Assoc):
            otherdict = other.dict()
            otherkeys = otherdict.keys()

            for key in selfkeys:
                selfval = selfdict[key]

                if key in otherkeys:
                    otherval = otherdict[key]

                    if (otherval not in null and selfval in null) \
                            or (otherval in null and selfval not in null) \
                            or (otherval not in null and selfval not in null and
                                selfval != otherval):
                        goodrows.append(key[0])
                        goodcols.append(key[1])
            for key in otherkeys:
                otherval = otherdict[key]

                if otherval not in null:
                    if key in selfkeys:
                        continue  # Already addressed
                    else:
                        goodrows.append(key[0])
                        goodcols.append(key[1])
        elif other not in null:
            for key in selfkeys:
                if selfdict[key] != other:
                    goodrows.append(key[0])
                    goodcols.append(key[1])
        else:
            for key in selfkeys:
                if selfdict[key] not in null:
                    goodrows.append(key[0])
                    goodcols.append(key[1])

        A = Assoc(goodrows, goodcols, 1)
        return A

    def __lt__(self, other):
        """
                    Element-wise strictly less than comparison between self and other.
                        Usage:
                            A < B
                        Input:
                            A = Associative Array
                            B = other object, e.g., another associative array, a number, or a string
                        Output:
                            A < B = An associative array such that for row and column labels r and c, resp., such that
                                    (A < B)(r,c) = 1 if and only if...
                                        (Case 1) A(r,c) < B(r,c) (when B is another associative array)
                                        (Case 2) A(r,c) < B (when B is not another associative array)
                                    otherwise (A < B)(r,c) = null.
                        Notes:
                            - Only numeric and string data types are supported.
                            - Any non-null string is always greater than null.
                            - If A(r,c) and B (or B(r,c) if B is another associative array) are incomparable
                            (e.g., if the former is a non-null number and the latter is a non-null string)
                            then an error is raised.
        """
        null = [0, '', None]

        selfdict = self.dict()
        selfkeys = self.dict().keys()
        goodrows = list()
        goodcols = list()

        if isinstance(other, Assoc):
            otherdict = other.dict()
            otherkeys = otherdict.keys()

            for key in selfkeys:
                selfval = selfdict[key]
                if key in otherkeys:
                    otherval = otherdict[key]

                    # Check if selfval and otherval have compatible types and whether selfval < otherval,
                    # where selfval and otherval are interpreted appropriately if null
                    if (selfval in null and ((is_numeric(otherval) and otherval > 0)
                                             or (isinstance(otherval, str) and otherval not in null)))\
                            or (otherval in null and is_numeric(selfval) and selfval < 0)\
                            or (is_numeric(otherval) and is_numeric(selfval) and selfval < otherval)\
                            or (isinstance(selfval, str) and isinstance(otherval, str) and selfval < otherval):
                        goodrows.append(key[0])
                        goodcols.append(key[1])
                else:
                    if is_numeric(selfval) and selfval < 0:
                        goodrows.append(key[0])
                        goodcols.append(key[1])
            for key in otherkeys:
                otherval = otherdict[key]

                if key in selfkeys:
                    continue  # Already addressed
                else:
                    if (isinstance(otherval, str) and otherval not in null) or (is_numeric(otherval) and 0 < otherval):
                        goodrows.append(key[0])
                        goodcols.append(key[1])
        else:
            for key in selfkeys:
                selfval = selfdict[key]

                # Check if selfval and other have compatible types and whether selfval < other,
                # where selfval and other are interpreted appropriately if null
                if (selfval in null and ((is_numeric(other) and other > 0)
                    or (isinstance(other, str) and other not in null)))\
                    or (other in null and is_numeric(selfval) and selfval < 0)\
                    or (is_numeric(other) and is_numeric(selfval) and selfval < other)\
                        or (isinstance(other, str) and isinstance(selfval, str) and selfval < other):

                    goodrows.append(key[0])
                    goodcols.append(key[1])

        A = Assoc(goodrows, goodcols, 1)
        return A

    def __gt__(self, other):
        """
                    Element-wise strictly greater than comparison between self and other.
                        Usage:
                            A > B
                        Input:
                            A = Associative Array
                            B = other object, e.g., another associative array, a number, or a string
                        Output:
                            A > B = An associative array such that for row and column labels r and c, resp., such that
                                    (A > B)(r,c) = 1 if and only if...
                                        (Case 1) A(r,c) > B(r,c) (when B is another associative array)
                                        (Case 2) A(r,c) > B (when B is not another associative array)
                                    otherwise (A > B)(r,c) = null.
                        Notes:
                            - Only numeric and string data types are supported.
                            - Any non-null string is always greater than null.
                            - If A(r,c) and B (or B(r,c) if B is another associative array) are incomparable
                            (e.g., if the former is a non-null number and the latter is a non-null string)
                            then an error is raised.
        """
        null = [0, '', None]

        selfdict = self.dict()
        selfkeys = self.dict().keys()
        goodrows = list()
        goodcols = list()

        if isinstance(other, Assoc):
            otherdict = other.dict()
            otherkeys = otherdict.keys()

            for key in selfkeys:
                selfval = selfdict[key]
                if key in otherkeys:
                    otherval = otherdict[key]

                    # Check if selfval and otherval have compatible types and whether selfval > otherval,
                    # where selfval and otherval are interpreted appropriately if null
                    if (selfval in null and is_numeric(otherval) and otherval < 0)\
                            or (otherval in null and ((is_numeric(selfval) and selfval > 0)
                                                      or (isinstance(selfval, str) and selfval not in null)))\
                            or (is_numeric(otherval) and is_numeric(selfval) and selfval > otherval)\
                            or (isinstance(otherval, str) and isinstance(otherval, str) and selfval > otherval):
                        goodrows.append(key[0])
                        goodcols.append(key[1])
                else:
                    if (is_numeric(selfval) and selfval > 0) or (isinstance(selfval, str) and selfval not in null):
                        goodrows.append(key[0])
                        goodcols.append(key[1])
            for key in otherkeys:
                otherval = otherdict[key]

                if key in selfkeys:
                    continue  # Already addressed
                else:
                    if is_numeric(otherval) and 0 > otherval:
                        goodrows.append(key[0])
                        goodcols.append(key[1])
        else:
            for key in selfkeys:
                selfval = selfdict[key]

                # Check if selfval and other have compatible types and whether selfval > other,
                # where selfval and other are interpreted appropriately if null
                if (selfval in null and is_numeric(other) and other < 0) \
                    or (other in null and ((is_numeric(selfval) and selfval > 0)
                        or (isinstance(selfval, str) and selfval not in null))) \
                    or (is_numeric(other) and is_numeric(selfval) and selfval > other) \
                        or (isinstance(other, str) and isinstance(other, str) and selfval > other):
                    goodrows.append(key[0])
                    goodcols.append(key[1])

        A = Assoc(goodrows, goodcols, 1)
        return A

    def __le__(self, other):
        """
                    Element-wise less than or equal comparison between self and other.
                        Usage:
                            A <= B
                        Input:
                            A = Associative Array
                            B = other object, e.g., another associative array, a number, or a string
                        Output:
                            A <= B = An associative array such that for row and column labels r and c, resp., such that
                                    (A <= B)(r,c) = 1 if and only if...
                                        (Case 1) A(r,c) <= B(r,c) (when B is another associative array)
                                        (Case 2) A(r,c) <= B (when B is not another associative array)
                                    otherwise (A <= B)(r,c) = null.
                        Notes:
                            - Only numeric and string data types are supported.
                            - Any non-null string is always greater than null.
                            - If A(r,c) and B (or B(r,c) if B is another associative array) are incomparable
                            (e.g., if the former is a non-null number and the latter is a non-null string)
                            then an error is raised.
        """
        null = [0, '', None]

        selfdict = self.dict()
        selfkeys = self.dict().keys()
        goodrows = list()
        goodcols = list()

        if isinstance(other, Assoc):
            otherdict = other.dict()
            otherkeys = otherdict.keys()

            for key in selfkeys:
                selfval = selfdict[key]
                if key in otherkeys:
                    otherval = otherdict[key]

                    # Check if selfval and otherval have compatible types and whether selfval <= otherval,
                    # where selfval and otherval are interpreted appropriately if null
                    if (selfval in null and ((is_numeric(otherval) and otherval >= 0)
                                             or isinstance(otherval, str)))\
                            or (otherval in null and is_numeric(selfval) and selfval <= 0)\
                            or (is_numeric(otherval) and is_numeric(selfval) and selfval <= otherval)\
                            or (isinstance(otherval, str) and isinstance(otherval, str) and selfval <= otherval):
                        goodrows.append(key[0])
                        goodcols.append(key[1])
                else:
                    if is_numeric(selfval) and selfval <= 0:
                        goodrows.append(key[0])
                        goodcols.append(key[1])
            for key in otherkeys:
                otherval = otherdict[key]

                if key in selfkeys:
                    continue  # Already addressed
                else:
                    if isinstance(otherval, str) or (is_numeric(otherval) and 0 <= otherval):
                        goodrows.append(key[0])
                        goodcols.append(key[1])
        else:
            for key in selfkeys:
                selfval = selfdict[key]

                # Check if selfval and other have compatible types and whether selfval <= other,
                # where selfval and other are interpreted appropriately if null
                if (selfval in null and ((is_numeric(other) and other >= 0)
                                         or isinstance(other, str))) \
                        or (other in null and is_numeric(selfval) and selfval <= 0) \
                        or (is_numeric(other) and is_numeric(selfval) and selfval <= other) \
                        or (isinstance(other, str) and isinstance(other, str) and selfval <= other):
                    goodrows.append(key[0])
                    goodcols.append(key[1])

        A = Assoc(goodrows, goodcols, 1)
        return A

    def __ge__(self, other):
        """
                    Element-wise greater than or equal comparison between self and other.
                        Usage:
                            A >= B
                        Input:
                            A = Associative Array
                            B = other object, e.g., another associative array, a number, or a string
                        Output:
                            A >= B = An associative array such that for row and column labels r and c, resp., such that
                                    (A >= B)(r,c) = 1 if and only if...
                                        (Case 1) A(r,c) >= B(r,c) (when B is another associative array)
                                        (Case 2) A(r,c) >= B (when B is not another associative array)
                                    otherwise (A >= B)(r,c) = null.
                        Notes:
                            - Only numeric and string data types are supported.
                            - Any non-null string is always greater than null.
                            - If A(r,c) and B (or B(r,c) if B is another associative array) are incomparable
                            (e.g., if the former is a non-null number and the latter is a non-null string)
                            then an error is raised.
        """
        null = [0, '', None]

        selfdict = self.dict()
        selfkeys = self.dict().keys()
        goodrows = list()
        goodcols = list()

        if isinstance(other, Assoc):
            otherdict = other.dict()
            otherkeys = otherdict.keys()

            for key in selfkeys:
                selfval = selfdict[key]
                if key in otherkeys:
                    otherval = otherdict[key]

                    # Check if selfval and otherval have compatible types and whether selfval >= otherval,
                    # where selfval and otherval are interpreted appropriately if null
                    if (selfval in null and is_numeric(otherval) and otherval <= 0)\
                            or (otherval in null and ((is_numeric(selfval) and selfval >= 0)
                                                      or isinstance(selfval, str)))\
                            or (is_numeric(otherval) and is_numeric(selfval) and selfval >= otherval)\
                            or (isinstance(otherval, str) and isinstance(otherval, str) and selfval >= otherval):
                        goodrows.append(key[0])
                        goodcols.append(key[1])
                else:
                    if (is_numeric(selfval) and selfval >= 0) or isinstance(selfval, str):
                        goodrows.append(key[0])
                        goodcols.append(key[1])
            for key in otherkeys:
                otherval = otherdict[key]

                if key in selfkeys:
                    continue  # Already addressed
                else:
                    if is_numeric(otherval) and 0 > otherval:
                        goodrows.append(key[0])
                        goodcols.append(key[1])
        else:
            for key in selfkeys:
                selfval = selfdict[key]

                # Check if selfval and other have compatible types and whether selfval >= other,
                # where selfval and other are interpreted appropriately if null
                if (selfval in null and is_numeric(other) and other <= 0) \
                        or (other in null and ((is_numeric(selfval) and selfval >= 0)
                            or isinstance(selfval, str))) \
                        or (is_numeric(other) and is_numeric(selfval) and selfval >= other) \
                        or (isinstance(other, str) and isinstance(other, str) and selfval >= other):
                    goodrows.append(key[0])
                    goodcols.append(key[1])

        A = Assoc(goodrows, goodcols, 1)
        return A


def val2col(array_: 'Assoc', separator: Optional[str] = None) -> 'Assoc':
    """Convert from adjacency array to incidence array.
            Usage:
                val2col(A, separator)
            Inputs:
                A = Associative Array
                separator = (new) delimiting character (default '|') to separate column labels from values
            Output:
                val2col(A, separator) = Associative Array B where B.row == A.row and
                                        B[row_label, col_label + split_separator + value] == 1 if and only if
                                        A[row_label, col_label] == value
    """
    if separator is None:
        separator = '|'

    rows, column_types, column_vals = array_.find()
    column_types = num_to_str(column_types)
    column_vals = num_to_str(column_vals)
    cols = catstr(column_types, column_vals, separator)
    return Assoc(rows, cols, 1)


def col2type(array_: 'Assoc', separator: Optional[str] = None) -> 'Assoc':
    """Split column keys of associative array and sorts first part as column key and second part as value.
        Inverse of val2col.
            Usage:
                B = col2type(A, separator)
            Inputs:
                A = Associative array with string column keys assumed to be of the form 'key' + separator + 'val'
                separator = separator for A's column keys (default '|')
            Outputs:
                col2type(A, separator) = Associative array whose row keys are the same as A, but whose column
                                        keys are the first parts of A's column keys and whose values are the second
                                        parts of A's column keys
            Example:
                col2type(A, '|')
                col2type(A, '/')
            Note:
                - A's column keys must be in the desired form.
    """
    if separator is None:
        separator = '|'

    # Extract row and column keys from A
    row, col, _ = array_.find()

    # Split column keys according to splitSep
    try:
        column_splits = [column_key.split(separator) for column_key in col]
    except:
        raise ValueError('Input column keys not of correct form.')

    # Extract column types and values
    column_types = [split_column_key[0] for split_column_key in column_splits]
    column_vals = [split_column_key[1] for split_column_key in column_splits]

    return Assoc(row, column_types, column_vals)


def readcsvtotriples(filename, labels=True, triples=False, **fmtoptions):
    """
    Read CSV file to row, col, val lists.
        Usage:
            row, col, val = readCSV(filename, labels=False, triples=False)
            row, col, val = readCSV(filename, fmtoptions)
        Inputs:
            filename = name of file (string)
            labels = optional parameter to say whether row and column labels are 
                        the first column and row, resp.
            triples = optional parameter indicating whether each row is of the form 'row[i], col[i], val[i]'
            **fmtoptions = format options accepted by csv.reader, e.g. "delimiter='\t'" for tsv's
        Outputs:
            row, col, val = value in row[i]-th row and col[i]-th column is val[i] (if not triples,
                                else the transposes of the columns)
        Examples:
            row, col, val = readcsv('my_file_name.csv')
            row, col, val = readcsv('my_file_name.csv', delimiter=';')
            row, col, val = readcsv('my_file_name.tsv', triples=True, delimiter='\t')
    """
    # Read CSV file and create (row-index,col-index):value dictionary
    with open(filename, 'rU') as csvfile:
        assoc_reader = csv.reader(csvfile, **fmtoptions)
        
        if triples:
            row = list()
            col = list()
            val = list()
            
            for line in assoc_reader:
                if len(line) == 0:
                    continue
                if len(line) != 3:
                    raise ValueError('line has ' + str(len(line)) + ' elements:\n' + str(line)
                                     + '\ntriples=True implies there are three columns')
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
                    raise ValueError("row has " + str(len(row)) + " elements while there are "
                                     + str(len(headings)) + " column labels.")
                else:
                    # If labels are expected, first element of row is row label, otherwise actual value
                    if labels:
                        start = 1
                    else:
                        start = 0

                    for i in range(start, len(row)):
                        # If labels are expected, use with dictionary
                        if row[i] is not None and row[i] != '':
                            if labels:
                                assoc_dict[(row[0], headings[i])] = row[i]
                            else:
                                assoc_dict[(line_num, i)] = row[i]
                # Increment line counter
                if not labels:
                    line_num += 1

            # Extract row, col, val from dictionary
            row_col_tuples = list(assoc_dict.keys())
            row = [str_to_num(item[0]) for item in row_col_tuples]
            col = [str_to_num(item[1]) for item in row_col_tuples]
            val = [str_to_num(item) for item in list(assoc_dict.values())]

    return row, col, val


def readcsv(filename, labels=True, triples=False, **fmtoptions):
    """Read CSV file to Assoc instance.
        Usage:
            A = readcsv(filename)
            A = readcsv(filename, fmtoptions)
        Inputs:
            filename = name of file (string)
            labels = optional parameter to say whether row and column labels are 
                        the first column and row, resp.
            triples = optional parameter indicating whether each row is of the form 'row[i], col[i], val[i]'
            fmtoptions = format options accepted by csv.reader, e.g. "delimiter='\t'" for tsv's
        Outputs:
            A = Associative Array whose column indices are given in the first line of the file,
                whose row indices are given in the first column of the file, and whose values
                are the remaining non-empty/null items, paired with the appropriate row
                and col indices
        Examples:
            A = readcsv('my_file_name.csv')
            A = readcsv('my_file_name.csv', delimiter=';')
            A = readcsv('my_file_name.tsv', delimiter='\t')
    """
    row, col, val = readcsvtotriples(filename, labels=labels, triples=triples, **fmtoptions)

    return Assoc(row, col, val)


def writecsv(array_, filename, fmtparams=None):
    """Write CSV file from Assoc instance.
        Usage:
            writeCSV(filename)
            writeCSV(filename, fmtoptions)
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
    with open(filename, 'w') as csv_file:
        assoc_writer = csv.writer(csv_file, fmtparams, lineterminator='\r')

        # Write the headings (offset by one to account for row indices)
        headings = [item for item in array_.col]
        headings.insert(0, None)
        assoc_writer.writerow(headings)

        # Create lookup dictionary
        row, col, val = array_.find()
        adj_dict = dict(zip(zip(row, col), val))

        for i in range(len(array_.row)):
            new_line = list()
            new_line.append(array_.row[i])  # Start with row index

            # Go through the row and add value if it exists, else None
            for j in range(len(array_.col)):
                if (array_.row[i], array_.col[j]) in adj_dict.keys():
                    new_line.append(adj_dict[(array_.row[i], array_.col[j])])
                else:
                    new_line.append(None)

            assoc_writer.writerow(new_line)

    return None
