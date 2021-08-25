# Import packages
from __future__ import annotations
import os
import warnings
from numbers import Number
import numpy as np
from typing import Union, Optional, List, Sequence, Callable, Tuple
from pkg_resources import resource_filename  # Facilitate packaged jars being referenced
import py4j.java_gateway

import D4M.assoc
import D4M.util

KeyVal = Union[str, Number]
StrList = Union[str, Sequence[str]]
ArrayLike = Union[KeyVal, Sequence[KeyVal], np.ndarray]
Selectable = Union[ArrayLike, slice, Callable]


# Classes
class JavaConnector:
    """Handles the starting of the JVM and exposing the JVM to the relevant JARs of Graphulo and Accumulo.
        Attributes:
            jvm_port = (Global) port number for port being used by py4j to connect to JVM
            jvm_gateway = (Global) py4j.java_gateway.JavaGateway object to facilitate communication with JVM
            default_paths = (Global) dictionary containing default paths for the py4j, accumulo, and graphulo JARs
                as well as path for directory containing /databases/[accumulo-db-instance] or path for config file
    """

    jvm_port = None
    jvm_gateway = None
    default_paths = {'py4j_path': resource_filename(__name__, '/jars/py4j/share/py4j/py4j.0.1.jar'),
                     'accumulo_path': resource_filename(__name__, '/jars/libext/*'),
                     'graphulo_path': resource_filename(__name__, '/jars/lib/graphulo-3.0.0.jar'),
                     'config': '/home/gridsan/tools/groups/'}

    @staticmethod
    def start_java(py4j_path: str = default_paths['py4j_path'], accumulo_path: str = default_paths['accumulo_path'],
                   graphulo_path: str = default_paths['graphulo_path'], die_on_exit: bool = True) \
            -> py4j.java_gateway.JavaGateway:
        """Start JVM and connect via a Py4J JavaGateway, with classpath exposing Accumulo and Graphulo to the JVM.
            Inputs:
                py4j_path = (Optional, default is packaged py4j jar) full path for py4j jar
                accumulo_path = (Optional, default is packaged accumulo jar) full path of _directory_ containing
                                Accumulo jars
                graphulo_path = (Optional, default is packaged graphulo_jar) full path for graphulo jar
                die_on_exit = (Optional, default True) Boolean whether JVM should close upon exiting python
            Output:
                start_java() = JavaConnector.gateway
            Notes:
                - Default values of py4j_path, accumulo_path, and graphulo_path are found in JavaConnector.default_paths
        """
        class_path = '.:' + accumulo_path + ':' + graphulo_path

        port = py4j.java_gateway.launch_gateway(jarpath=py4j_path, classpath=class_path, die_on_exit=die_on_exit)

        gateway = py4j.java_gateway.JavaGateway(
            gateway_parameters=py4j.java_gateway.GatewayParameters(port=port),
            callback_server_parameters=py4j.java_gateway.CallbackServerParameters(port=0)
        )

        print('JavaGateway started in Port '+str(port))
        
        JavaConnector.jvm_port = port
        JavaConnector.jvm_gateway = gateway
        return gateway

    @staticmethod
    def get_port() -> int:
        """Return JavaGateway port number."""
        return JavaConnector.jvm_port
    
    @staticmethod
    def get_gateway() -> py4j.java_gateway.JavaGateway:
        """Return the JavaGateway instance."""
        return JavaConnector.jvm_gateway


class DbServer:
    """A DbServer instance collects the data needed to bind to Accumulo tables in a given Accumulo instance.
        Attributes:
            instance_name = name of the Accumulo instance, i.e., the name of the Accumulo instance directory
            host = hostname associated with the Accumulo instance
            user = username associated with the Accumulo instance
            password = password associated with the Accumulo instance
            db_type = type of Accumulo database (e.g., BigTableLike)
            gateway = py4j.java_gateway.JavaGateway object
        Note:
            - instance_name, host, user, and password are used as arguments of edu.mit.ll.d4m.db.cloud.D4mDbInfo
    """

    def __init__(self, instance_name: str, host: str, user: str, password: str, db_type: str,
                 gateway: py4j.java_gateway.JavaGateway):
        """Construct DbServer instance from database connection information."""
        self.instance_name = instance_name
        self.host = host
        self.user = user
        self.password = password
        self.db_type = db_type
        self.gateway = gateway

    def ls(self) -> List[str]:
        """Print list of tables in DbServer instance."""
        db_info = self.gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDbInfo(self.instance_name, self.host, self.user,
                                                                     self.password)
        tables = db_info.getTableList().split(' ')
        tables.pop()
        return tables


class DbTable:
    """Binding information to existing table.
        Attributes:
            DB = DbServer containing table
            name = table name
            security = authorization argument for edu.mit.ll.d4m.db.cloud.D4mDataSearch; usually ''
            num_limit = maximum size of batch of triples fetched by successive calls of
                edu.mit.ll.d4m.db.cloud.D4mDataSearch.next()
            num_row
            column_family
            put_bytes = upper bound on number of bytes ingested at a time
            d4m_query = py4j.java_gateway.JavaObject binding edu.mit.ll.d4m.db.cloud.D4mDataSearch
            table_ops = py4j.java_gateway.JavaObject binding edu.mit.ll.d4m.db.cloud.D4mDbTableOperations
    """

    def __init__(self, DB: 'DbServer', name: str, security: str, num_limit: int, num_row: int,
                 column_family: str, put_bytes: float, d4m_query: py4j.java_gateway.JavaObject,
                 table_ops: py4j.java_gateway.JavaObject):
        self.DB = DB
        self.name = name
        self.security = security
        self.num_limit = num_limit
        self.num_row = num_row
        self.column_family = column_family
        self.put_bytes = put_bytes
        self.d4m_query = d4m_query
        self.table_ops = table_ops


class DbTablePair:
    """Binding information to existing pair of tables.
        Attributes:
            DB = DbServer containing table
            name_1 = name of first table in pair
            name_2 = name of second table in pair
            security = authorization argument for edu.mit.ll.d4m.db.cloud.D4mDataSearch; usually ''
            num_limit = maximum size of batch of triples fetched by successive calls of
                edu.mit.ll.d4m.db.cloud.D4mDataSearch.next()
            num_row
            column_family
            put_bytes = upper bound on number of bytes ingested at a time
            d4m_query = py4j.java_gateway.JavaObject binding edu.mit.ll.d4m.db.cloud.D4mDataSearch
            table_ops = py4j.java_gateway.JavaObject binding edu.mit.ll.d4m.db.cloud.D4mDbTableOperations
    """

    def __init__(self, DB: 'DbServer', name_1: str, name_2: str, security: str, num_limit: int,
                 num_row: int, column_family: str, put_bytes: float, d4m_query: py4j.java_gateway.JavaObject,
                 table_ops: py4j.java_gateway.JavaObject):
        self.DB = DB
        self.name_1 = name_1
        self.name_2 = name_2
        self.security = security
        self.num_limit = num_limit
        self.num_row = num_row
        self.column_family = column_family
        self.put_bytes = put_bytes
        self.d4m_query = d4m_query
        self.table_ops = table_ops


DbTableLike = Union[DbTable, DbTablePair]
default_tables = ['accumulo.metadata', 'accumulo.replication', 'accumulo.root', 'trace']


def dbsetup(instance: str, config: str = JavaConnector.default_paths['config'], py4j_path: Optional[str] = None,
            accumulo_path: Optional[str] = None, graphulo_path: Optional[str] = None, die_on_exit: bool = True,
            force_restart: bool = False) -> 'DbServer':
    """Set up DB connection, starting JVM if not already started.
        Usage:
            dbsetup('instance_name')
            dbsetup('instance_name', config='config_location')
        Input:
            instance = name of existing accumulo instance
            config = location of config information
                        -- if a directory then the file
                        accumulo_user_password.txt is opened in config/databases/instance
                        to get password, and dnsname is open in config/databases/instance
                        for hostname. Username is assumed to be 'AccumuloUser'
                        -- if a file, then it is assumed to be of form
                            "instance = *actual instance name*
                             hostname = *actual hostname*
                             username = *actual username*
                             password = *actual password*"
            py4j_path, accumulo_path, graphulo_path, die_on_exit = see JavaConnector.start_java()
        Output:
            DB = Dbserver, containing the connection information to Accumulo instance
        Examples:
            dbsetup('class-db49')
            dbsetup('class-db50')
    """
    if os.path.isdir(config):
        dbdir = config + '/databases/' + instance
        with open(dbdir + '/accumulo_user_password.txt', 'r') as f:
            pword = f.read()
        with open(dbdir + '/dnsname', 'r') as f:
            hostname = f.read()
            hostname = hostname.replace('\n', '') + ':2181'
            username = "AccumuloUser"
    elif os.path.isfile(config):
        with open(config, 'r') as f:
            conf = [line.split('=') for line in f.readlines()]
            conf = {line[0]: line[1] for line in conf}
        instance = conf['instance']
        hostname = conf['hostname']
        username = conf['username']
        pword = conf['password']
    else:
        raise ValueError("'config' must either be a config file or a directory containing a config file")

    if JavaConnector.jvm_gateway is None or force_restart:
        gateway = JavaConnector.start_java(py4j_path=py4j_path, accumulo_path=accumulo_path,
                                           graphulo_path=graphulo_path, die_on_exit=die_on_exit)
    else:
        gateway = JavaConnector.jvm_gateway

    return DbServer(instance, hostname, username, pword, "BigTableLike", gateway)


def _get_index_single(DB: 'DbServer', table_name: str, security: str = '', num_limit: int = 0, num_row: int = 0,
                      column_family: str = '', put_bytes: float = 5e5) -> 'DbTable':
    """Create DbTable object containing binding information for tableName in DB."""
    gateway = DB.gateway
    table_ops = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDbTableOperations
    table_ops_object = table_ops(DB.instance_name, DB.host, DB.user, DB.password)

    if table_name not in DB.ls():
        print("Creating " + table_name + " in " + DB.instance_name + ".")
        table_ops_object.createTable(table_name)

    data_search = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDataSearch
    query_object = data_search(DB.instance_name, DB.host, table_name, DB.user, DB.password)

    db_table = DbTable(DB, table_name, security, num_limit, num_row, column_family, put_bytes,
                       query_object, table_ops_object)
    return db_table


def _get_index_pair(DB: 'DbServer', table_name_1: str, table_name_2: str, security: str = '', num_limit: int = 0,
                    num_row: int = 0, column_family: str = '', put_bytes: float = 5e5) -> 'DbTablePair':
    """Create DbTablePair object containing binding information for tableName1 and tableName2 in DB."""
    gateway = DB.gateway
    table_ops = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDbTableOperations
    table_ops_object = table_ops(DB.instance_name, DB.host, DB.user, DB.password)
    for table_name in [table_name_1, table_name_2]:
        if table_name not in DB.ls():
            print("Creating " + table_name + " in " + DB.instance_name + ".")
            table_ops_object.createTable(table_name)

    data_search = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDataSearch
    query_object = data_search(DB.instance_name, DB.host, table_name_1, DB.user, DB.password)

    db_table_pair = DbTablePair(DB, table_name_1, table_name_2, security, num_limit, num_row, column_family, put_bytes,
                                query_object, table_ops_object)
    return db_table_pair


def _get_index_assoc(table: DbTableLike, row_query: Selectable, col_query: Selectable) -> D4M.assoc.Assoc:
    """Create Assoc object from the sub-array of table with queried row and column indices."""
    switch_keys = False
    if isinstance(table, DbTablePair):
        if row_query == ':' and col_query != ':':
            table.d4m_query.setTableName(table.name_2)
            switch_keys = True
        else:
            table.d4m_query.setTableName(table.name_1)
    else:
        table.d4m_query.setTableName(table.name)

    table.d4m_query.setCloudType(table.DB.db_type)
    table.d4m_query.setLimit(table.num_limit)

    row_query_needs_full = not ((isinstance(row_query, str) and ':' not in row_query) or row_query == ':')
    col_query_needs_full = not ((isinstance(col_query, str) and ':' not in col_query) or col_query == ':')

    if row_query_needs_full or col_query_needs_full:
        table.d4m_query.reset()
        table.d4m_query.doMatlabQuery(':', ':', table.column_family, table.security)
        full_row, full_col = table.d4m_query.getRowReturnString(), table.d4m_query.getColumnReturnString()

        row_query, col_query = D4M.util.select_items(row_query, full_row), D4M.util.select_items(col_query, full_col)
        row_query, col_query = D4M.util.to_db_string(row_query), D4M.util.to_db_string(col_query)
    else:
        if row_query != ':':
            row_query = D4M.util.to_db_string(row_query)
        if col_query != ':':
            col_query = D4M.util.to_db_string(col_query)

    table.d4m_query.reset()
    table.d4m_query.doMatlabQuery(row_query, col_query, table.column_family, table.security)
    new_row, new_col = table.d4m_query.getRowReturnString(), table.d4m_query.getColumnReturnString()
    new_val = table.d4m_query.getValueReturnString()

    if switch_keys:
        new_row, new_col = new_col, new_row

    return D4M.assoc.Assoc(new_row, new_col, new_val)


def _get_index_from_iter(table: 'DbTableLike') -> 'D4M.assoc.Assoc':
    """Query table as iterator if table.num_limit > 0 and return associative array consisting of next batch of triples,
        otherwise return associative array consisting of all triples."""
    if table.num_limit == 0:
        return get_index(table, ':', ':')

    table_name = table.d4m_query.getTableName()

    if isinstance(table, DbTablePair) and table_name == table.name_2:
        col = table.d4m_query.getRowReturnString()
        row = table.d4m_query.getColumnReturnString()
    else:
        row = table.d4m_query.getRowReturnString()
        col = table.d4m_query.getColumnReturnString()

    val = table.d4m_query.getValueReturnString()

    if not table.d4m_query.hasNext():
        warnings.warn('End of table reached.')

    table.d4m_query.next()

    return D4M.assoc.Assoc(row, col, val)


def get_index(*arg) -> Union[DbTableLike, D4M.assoc.Assoc]:
    """Query a table in a given Accumulo database instance, either returning a DbTable/DbTablePair object or an
        associative array containing triples from the given table.
        Inputs:
            *arg = one of:
                - DB, table_name - a DbServer instance and the name of a table (which may or may not be the
                    name of an existing table)
                - DB, table_name_1, table_name_2 - a DbServer instance and the name of two tables (which each
                    may or may not be the name of an existing table)
                - table, row_query, col_query - a DbTable or DbTablePair with a pair of objects representing
                    row and column queries, respectively (see D4M.util.select_items for valid & supported queries)
                - table - a DbTable or DbTablePair
        Output:
            get_index(DB, table_name) = DbTable with binding information for table_name, if it exists,
                otherwise creates a table in DB with default settings, and binds to that table
            get_index(DB, table_name_1, table_name_2) = DBtable with binding information for tableName1 and tablename2,
                if they exist, otherwise creates necessary tables in DB with default settings, and binds to those tables
            get_index(table, row_query, col_query) = associative array containing all triples in table whose row and
                column keys are selected by the given row_query and col_query, respectively
            get_index(table) = if table.num_limit > 0, table is interpreted as an iterator and an associative array
                containing the next batch of triples; if table.num_limit == 0, mirrors get_index(table, ':', ':')
    """
    if len(arg) == 2 and isinstance(arg[0], DbServer) and isinstance(arg[1], str):
        DB, table_name = arg
        output = _get_index_single(DB, table_name)
    elif len(arg) == 3 and isinstance(arg[0], DbServer) and isinstance(arg[1], str) and isinstance(arg[2], str):
        DB, table_name_1, table_name_2 = arg
        output = _get_index_pair(DB, table_name_1, table_name_2)
    elif len(arg) == 3 and (isinstance(arg[0], DbTable) or isinstance(arg[0], DbTablePair)):
        # Assume arg[1] and arg[2] can be interpreted as row and column queries, respectively
        table, row_query, column_query = arg
        output = _get_index_assoc(table, row_query, column_query)
    elif len(arg) == 1 and (isinstance(arg[0], DbTable) or isinstance(arg[0], DbTablePair)):
        table = arg[0]
        output = _get_index_from_iter(table)
    else:
        raise ValueError("Improper argument supplied. Argument must be of the form 'DB, table_name', "
                         "'DB, table_name_1, table_name_2', 'table, row_query, col_query', or 'table'.")
    return output


valid_confirm = ['y', 'yes', 'Y', 'Yes']  # Enumerate user inputs that 'confirm'


def _delete_table_single(table: 'DbTable', force: bool = False) -> None:
    """Delete the table with name table.name in table.DB instance."""
    if table.name in table.DB.ls():
        if table.name in default_tables:
            print(table.name + ' is a default table and cannot be deleted.')
        else:
            if force:
                confirm = 'yes'
            else:
                confirm = input("Confirm deletion of " + table.name + " in " + table.DB.instance_name + ".")
            if confirm in valid_confirm:
                table.table_ops.deleteTable(table.name)
                print("Deleted " + table.name + " from " + table.DB.instance_name + ".")
    else:
        print(table.name + " was not found in " + table.DB.instance_name + ".")
    return None


def _delete_table_pair(table_pair: 'DbTablePair', force: bool = False) -> None:
    """Delete the tables with names table.name1 and table.name2 in table.DB instance."""
    present_tables = table_pair.DB.ls()
    table_pair_names = [table_pair.name_1, table_pair.name_2]
    is_present_list = [table_pair.name_1 in present_tables, table_pair.name_2 in present_tables]
    is_default_list = [table_pair.name_1 in default_tables, table_pair.name_2 in default_tables]

    for table_index in range(2):
        is_present = is_present_list[table_index]
        is_default = is_default_list[table_index]
        table_name = table_pair_names[table_index]
        if is_present and not is_default:
            if force:
                confirm = 'yes'
            else:
                confirm = input('Confirm deletion of ' + table_name + ' in ' + table_pair.DB.instance_name + '.')
            if confirm in valid_confirm:
                table_pair.table_ops.deleteTable(table_name)
                print('Deleted ' + table_name + ' from ' + table_pair.DB.instance_name + '.')
        elif is_present and is_default:
            print(table_name + ' is a default table and cannot be deleted.')
        else:
            print(table_name + " was not found in " + table_pair.DB.instance_name + ".")
    return None


def delete_table(table: DbTableLike, force: bool = False) -> None:
    """Delete given DbTable or DbTablePair from table.DB.
        Notes:
            - Deletes ACTUAL table from table.DB.
            - Default Accumulo tables cannot be deleted.
            - When deleting DbTablePair, deletion of table.table_name_1 and table.table_name_2 are confirmed separately.
    """
    if isinstance(table, DbTablePair):
        _delete_table_pair(table, force=force)
    else:
        _delete_table_single(table, force=force)
    return None


def delete_all(DB: DbServer, force: bool = False) -> None:
    """Deletes all (non-default) tables in the given DB.
        Notes:
            - Deletes the ACTUAL non-default tables from DB.
    """
    present_tables = DB.ls()
    if force:
        confirm = 'yes'
    else:
        confirm = input('Confirm deletion of all non-default tables in ' + DB.instance_name + '.')
    if confirm in valid_confirm:
        for table_name in present_tables:
            temp_table = get_index(DB, table_name)
            if table_name not in default_tables:
                delete_table(temp_table, force=True)
    return None


def get_iterator(table: DbTableLike, num_limit: int) -> DbTableLike:
    """Query iterator functionality."""
    if isinstance(table, DbTable):
        table_iter = DbTable(table.DB, table.name, table.security, num_limit, table.num_row, table.column_family,
                             table.put_bytes, table.d4m_query, table.table_ops)
    else:
        table_iter = DbTablePair(table.DB, table.name_1, table.name_2, table.security, num_limit, table.num_row,
                                 table.column_family, table.put_bytes, table.d4m_query, table.table_ops)

    table_iter.d4m_query.setCloudType(table_iter.DB.db_type)
    table_iter.d4m_query.setLimit(num_limit)
    table_iter.d4m_query.reset()
    table_iter.d4m_query.doMatlabQuery(':', ':', table_iter.column_family, table_iter.security)

    return table_iter


def put_triple(table: DbTableLike, row: ArrayLike, col: ArrayLike, val: ArrayLike) -> None:
    """Insert the triples (row(i), col(i), val(i)) into table.
        Usage:
            putTriple(table, row, col, val)
        Inputs:
            table = DBtable or DbTablePair instance
            row = string of (delimiter separated) values (delimiter is last character)
                or list of values of length n
            col = string of (delimiter separated) values (delimiter is last character)
                or list of values of length n
            val = string of (delimiter separated) values (delimiter is last character)
                or list of values of length n
        Notes:
            - If table is a DbTablePair, then table.name2 is assumed to be the transpose of
                table.name1 for the purposes or insertion (e.g., transposed triples are put
                into table.name2)
            - Accumulo tables record duplicate triples.
    """
    gateway = table.DB.gateway

    new_row = D4M.util.num_to_str(row)
    new_col = D4M.util.num_to_str(col)
    new_val = D4M.util.num_to_str(val)

    # Optimize by selecting appropriate chunk size for insertion
    chunk_bytes = table.put_bytes
    num_triples = len(row)
    avg_byte_per_triple = (np.char.str_len(new_row).sum() + np.char.str_len(new_col).sum()
                           + np.char.str_len(new_val).sum()) / num_triples
    chunk_size = int(min(max(1, np.round(chunk_bytes / avg_byte_per_triple)), num_triples))

    db_insert = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDbInsert
    if isinstance(table, DbTablePair):
        insert_obj = db_insert(table.DB.instance_name, table.DB.host, table.name_1, table.DB.user, table.DB.password)
        insert_objT = db_insert(table.DB.instance_name, table.DB.host, table.name_2, table.DB.user, table.DB.password)
    else:
        insert_obj = db_insert(table.DB.instance_name, table.DB.host, table.name, table.DB.user, table.DB.password)
        insert_objT = None

    for chunk_start in np.arange(0, num_triples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_triples)
        new_row_chunk = new_row[chunk_start:chunk_end]
        new_col_chunk = new_col[chunk_start:chunk_end]
        new_val_chunk = new_val[chunk_start:chunk_end]

        insert_obj.doProcessing(new_row_chunk, new_col_chunk, new_val_chunk, table.column_family, table.security)

        if isinstance(table, DbTablePair):
            insert_objT.doProcessing(new_col_chunk, new_row_chunk, new_val_chunk, table.column_family, table.security)
    return None


def put(table: DbTableLike, A: D4M.assoc.Assoc) -> None:
    """Put Assoc object A into table."""
    row, col, val = A.find()
    put_triple(table, row, col, val)
    return None


def nnz(table: DbTableLike) -> int:
    """Returns the number of non-zero triples stored in table.
        Note:
            - returns count for all triples found, including duplicates.
    """
    if isinstance(table, DbTablePair):
        table_name = table.name_1
    else:
        table_name = table.name

    gateway = table.DB.gateway
    table_list = gateway.jvm.java.util.ArrayList()
    table_list.add(table_name)

    nnz_ = table.table_ops.getNumberOfEntries(table_list)
    return nnz_


def add_splits(table: DbTableLike, split_string: str, split_stringT: Optional[str] = None) -> None:
    if isinstance(table, DbTable):
        table.table_ops.addSplits(table.name, split_string)
    else:
        assert isinstance(table, DbTablePair)
        if split_stringT is None:
            split_stringT = split_string
        table.table_ops.addSplits(table.name_1, split_string)
        table.table_ops.addSplits(table.name_2, split_stringT)
    return None


def get_splits(table: DbTableLike) -> Union[str, Tuple[str, str]]:
    if isinstance(table, DbTable):
        split_string = table.table_ops.getSplitsString(table.name)
        return split_string
    else:
        assert isinstance(table, DbTablePair)
        split_string = table.table_ops.getSplitsString(table.name_1)
        split_stringT = table.table_ops.getSplitsString(table.name_2)
        return split_string, split_stringT
