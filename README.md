# Python D4M

D4M.py is a module for Python that allows unstructured data to be represented as triples in sparse matrics (Associative Arrays) and can be manipulated using standard linear algebraic operations.
Using D4M it is possible to construct advanced analytics with just a few lines of code.
D4M was initially developed in MATLAB by Dr Jeremy Kepner and his team at Lincoln Laboratory. 
The goal is to implement D4M in a native Python method.

The D4M Project Page: <https://d4m.mit.edu/>

Current Status: Many of the functionalities of cores D4M have been implemented, and basic Accumulo/Graphulo connection capabilities exist.

## Documentation

- See below for installation brief use instructions.
- For examples of more extensive use, see the examples.
- The MATLAB D4M distribution contains an eight lecture course in its documentation (<https://github.com/Accla/d4m> in d4m/docs). Many of the examples from this course have been translated to Python and the concepts are relevant to both the MATLAB and Python versions, and so this course could serve as an introduction to D4M.py as well.
- When citing D4M in puplications, please use:
  - [Kepner et al, ICASSP 2012] Dynamic Distributed Dimensional Data Model (D4M) Database and Computation System, J. Kepner, W. Arcand, W. Bergeron, N. Bliss, R. Bond, C. Byun, G. Condon, K. Gregson, M. Hubbell, J. Kurz, A. McCabe, P. Michaleas, A. Prout, A. Reuther, A. Rosa & C. Yee, ICASSP (International Conference on Acoustics, Speech, and Signal Processing), Special session on Signal and Information Processing for "Big Data" (organizers: Bliss & Wolfe), March 25-30, 2012, Kyoto, Japan
  
## Requirements

D4M.py is written and tested to work with Python 3.6 through 3.9 . It makes use of
- Scipy.sparse (for sparse matrix support)
- Numpy (for working with scipy)
- Py4J (for handling database connectivity)

## Installation

You can clone D4M.py and in the D4M directory run 
```
python setup.py install
```
from the command line.

## Basic Use

Start by importing the D4M.py package:
```python
import D4M.assoc
```

Associative Arrays can be constructed from strings, arrays of strings, scalars, and arrays of numbers as row keys, column keys, and values:
```python
row = "a,a,a,a,a,a,a,aa,aaa,b,bb,bbb,a,aa,aaa,b,bb,bbb,"
column = "a,aa,aaa,b,bb,bbb,a,a,a,a,a,a,a,aa,aaa,b,bb,bbb,"
values = "a-a,a-aa,a-aaa,a-b,a-bb,a-bbb,a-a,aa-a,aaa-a,b-a,bb-a,bbb-a,a-a,aa-aa,aaa-aaa,b-b,bb-bb,bbb-bbb,"

A = D4M.assoc.Assoc(row, column, values)
```

You can get particular rows and columns of associative arrays by using row and column indexing, as well as get the entries where the values satisfy some condition.
```python
Ar = A["a,b,", :] # Select sub-associative array of A consisting only of rows "a" and "b"
Ac = A[:, "a,b,"] # Select sub-associative array of A consisting only of columns 'a' and "b"
Av = A > "b"      # 0,1-valued associative array corresponding to entries of A with value > "b"
```

Associative Arrays support a variety of mathematical operations, including addition, subtraction, matrix multiplication, element-wise multiplication/division, summing across rows/columns, and more.
```python
A + B         # Associative array addition
A - B         # Associative array difference
A * B         # Associative array multiplication
A.multiply(B) # Associative array element-wise multiplication
A.divide(B)   # Associative array element-wise division
A.sum()       # Sum over entire associative array
A.sum(0)      # Sum down columns of associative array
A.sum(1)      # Sum across rows of associative array
```

To support commonly used queries and operations, the D4M.util submodule contains additional utilities for working with D4M.assoc and D4M.db. Some example uses:
```python
import D4M.util

D4M.util.startswith("a,")       # Callable which, applied to a sequence of strings, returns indices of those starting with "a"
D4M.util.contains("a,")         # Callable which, applied to a sequence of strings, returns indices of those containing "a"
D4M.util.to_db_string("a,b,c,") # Properly formatted string for Accumuldo DB queries
```

For more exmaples of how you can use D4M.py, check out the examples in the examples directory, including some examples with real datasets. 

## Database Use

Use of the database connection capabilities requires Graphulo. Graphulo is available on this page: <https://github.com/Accla/graphulo.> To use, start by importing the D4M.db submodule:
```python
import D4M.db
```

D4M.py relies on the Py4J package to call the Graphulo functions that enable database communication. The command D4M.db.dbsetup(...) is used to connect to a Accumulo instance and automatically starts the JVM. Note that multiple calls of D4M.db.dbsetup(...) in a single session may create multiple JVM instances.

## Testing

***Note***
Various parts of this implementation have been completed and compared with the original matlab in performance. In the examples/Scaling subfolder, this implementation has achieved performance on par with the Julia and MATLAB implementations of D4M.

The associative array and utility submodules can be tested for correctness by running the command "pytest" in the D4M/test subdirectory. The database submodule can be included in this testing by renaming the file "\_test_db.py" to "test_db.py"; custom configuration info for py4j, Graphulo, Accumulo, and path to Accumulo instances may be included in "test_db_config.txt" for use when running tests.
