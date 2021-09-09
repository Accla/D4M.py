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

D4M.py relies on the Py4J package to call the Graphulo functions that enable database communication. The main setup command is `D4M.db.dbsetup(instance, **config_params)`, which returns a `D4M.db.DbServer` object containing configuration data for the indicated Accumulo instance as well as starting the JVM. 
- `instance` is the name of the Accumulo instance being connected to.
- `config_params` includes optional parameters:
  - `py4j_path`, `accumulo_path`, and `graphulo_path` are paths to the Py4J, Accumulo, and Graphulo JARs, respectively; the default paths use the packaged instances of these JARs, found in /D4M/jars/. 
  - `config` is a path to the Accumulo instance's configuration info, either a directory or a file. 
    If a directory, then the username is assumed to be 'AccumuloUser' and the password and hostname are found in files [config]/databases/[instance]/accumulo_user_password.txt and [config]/databases/[instance]/dnsname, respectively. 
    If a file, it is assumed to have lines of the form "instance=*actual instance name*", "hostname=*actual hostname*", "username=*actual username*", and "password=*actual password*". The default is for MIT Supercloud.
  - `force_restart` is a Boolean indicating whether the JVM should be restarted even if an existing one can be found. (Default is False.)
  - `die_on_exit` is a Boolean indicating whether the JVM should be shutdown when Python is closed. (Default is True.)

The main D4M.db commands:

```python
DB = D4M.db.dbsetup("test-db")  # Create DbServer object to connect to "test-db" Accumulo instance

Db.ls() # Print out names of tables currently in "test-db"

# Create DbTable binding to table "test" in "test-db" 
# or create table "test" in "test-db" and bind to it if no such table exists
test_table = D4M.db.get_index(DB, "test")   

# Create DbTablePair binding to tables "testpair" and "testpairT" in "test-db"
# or create tables "testpair" and/or "testpairT" in "test-db" and bind to them if no such table(s) exist
test_table_pair = D4M.db.get_index(DB, "testpair", "testpairT")

# Create DbTable which iterates through test_table one hundred entries at a time
test_iter = D4M.db.get_iterator(test_table, 100)

# Insert triples ('a', 'A', 'aA'), ('b', 'A', 'bA'), and ('b', 'B', 'bB') into table "test"
D4M.db.put_triple(test_table, "a,b,b,", "A,A,B,", "aA,bA,bB,")  # Method 1
test_table.put_triple("a,b,b,", "A,A,B,", "aA,bA,bB,")          # Method 2

# Insert triples ('a', 'A', 'aA'), ('b', 'A', 'bA'), and ('b', 'B', 'bB') into table "testpair"
# and ('A', 'a', 'aA'), ('A', 'b', 'bA'), and ('B', 'b', 'bB') into table "testpairT"
D4M.db.put_triple(test_table_pair, "a,b,b,", "A,A,B,", "aA,bA,bB,") # Method 1
test_table_pair.put_triple("a,b,b,", "A,A,B,", "aA,bA,bB,")         # Method 2

# Query tables
A = D4M.db.get_index(test_table)                  # Output table "test" as a D4M.assoc.Assoc
B = D4M.db.get_index(test_table, "a,", ":")       # Output row "a" of table "test" as a D4M.assoc.Assoc
C = D4M.db.get_index(test_table_pair, "a,", ":")  # Output row "a" of table "testpair" as a D4M.assoc.Assoc
D = D4M.db.get_index(test_table_pair, ":", "A,")  # Output row "A" of table "testpairT" as a D4M.assoc.Assoc
E = D4M.db.get_index(test_iter)                   # Output first hundred entries of table "test" as a D4M.assoc.Assoc
F = D4M.db.get_index(test_iter)                   # Output second hundred entries of table "test" as a D4M.assoc.Assoc

# Delete tables
D4M.db.delete_table(DB, test_table)                   # Delete table "test" from "test-db" after confirm
D4M.db.delete_table(DB, test_table, force=True)       # Delete table "test" from "test-db" without confirm
D4M.db.delete_table(DB, test_table_pair)              # Delete tables "testpair", "testpairT" from "test-db" after confirm
D4M.db.delete_table(DB, test_table_pair, force=True)  # Delete tables "testpair", "testpairT" from "test-db" without confirm
D4M.db.delete_all(DB)                                 # Delete all non-default tables from "test-db" after confirm
D4M.db.delete_all(DB, force=True)                     # Delete all non-default tables from "test-db" without confirm
```

## Testing

***Note***
Various parts of this implementation have been completed and compared with the original matlab in performance. In the examples/Scaling subfolder, this implementation has achieved performance on par with the Julia and MATLAB implementations of D4M.

The associative array and utility submodules can be tested for correctness by running the command "pytest" in the D4M/test subdirectory. The database submodule can be included in this testing by renaming the file "\_test_db.py" to "test_db.py"; custom configuration info for py4j, Graphulo, Accumulo, and path to Accumulo instances may be included in "test_db_config.txt" for use when running tests.
