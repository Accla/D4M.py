# Import packages
import getpass
import os
from numbers import Number
import warnings
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


valid_paths = ["py4j_path", "accumulo_path", "graphulo_path", "jtds_path", "postgresql_path", "mysql_path"]


default_paths = {"py4j_path": resource_filename(__name__, "/jars/py4j/share/py4j/py4j0.10.7.jar"),
                 "accumulo_path": resource_filename(__name__, "/jars/accumulo/libext/*"),
                 "graphulo_path": resource_filename(__name__, "/jars/graphulo/lib/graphulo-3.0.0.jar"),
                 "jtds_path": resource_filename(__name__, "/jars/jtds/jtds-1.3.1-dist/jtds-1.3.1.jar"),
                 "postgresql_path": resource_filename(__name__, "/jars/postgresql/postgresql-42.2.24.jar"),
                 "mysql_path": resource_filename(__name__, "/jars/mysql/mysql-connector-java-8.0.26/"
                                                           "mysql-connector-java-8.0.26.jar")
                 }


def _read_config_file(file: str) -> dict:
    with open(file, "r") as config_file:
        db_conf = [line.split("=") for line in config_file.readlines()]
        db_conf = {line[0]: line[1] for line in db_conf}
    return db_conf


def _get_default_path(
    field_name: str, default: str, file: Optional[str] = None, silent: bool = True
) -> str:
    if file is not None:
        try:
            db_conf = _read_config_file(file)
            try:
                return db_conf[field_name]
            except KeyError:
                message = "Given file has no field '" + str(field_name) + "'."
                if not silent:
                    raise ValueError(message)
                else:
                    print(message)
        except FileNotFoundError:
            message = "No file named '" + str(file) + "' found."
            if not silent:
                raise ValueError(message)
            else:
                print(message)
    return default


# Classes
class JavaConnector:
    """Handles the starting of the JVM and exposing the JVM to the relevant JARs of Graphulo and Accumulo.
    Attributes:
        jvm_port = (Global) port number for port being used by py4j to connect to JVM
        jvm_gateway = (Global) py4j.java_gateway.JavaGateway object to facilitate communication with JVM
        py4j_path = path to py4j0.x.jar
        accumulo_path = path to directory containing accumulo jars
        graphulo_path = path to graphulo-3.0.0.jar
        jtds_path = path to jtds-1.3.1.jar
    """

    jvm_port = None
    jvm_gateway = None
    paths = {path_name: default_paths[path_name] for path_name in valid_paths}
    py4j_path = default_paths["py4j_path"]
    accumulo_path = default_paths["accumulo_path"]
    graphulo_path = default_paths["graphulo_path"]
    jtds_path = default_paths["jtds_path"]
    postgresql_path = default_paths["postgresql_path"]
    mysql_path = default_paths["mysql_path"]

    @staticmethod
    def start_java(
        filename: Optional[str] = None,
        die_on_exit: bool = True,
        **paths
    ) -> py4j.java_gateway.JavaGateway:
        """Start JVM and connect via a Py4J JavaGateway, with classpath exposing Accumulo and Graphulo to the JVM.
        Inputs:
            py4j_path = (Optional, default is packaged py4j jar unless filename is given) full path for py4j jar
            accumulo_path = (Optional, default is packaged accumulo jar unless filename is given) full path of
                _directory_ containing Accumulo jars
            graphulo_path = (Optional, default is packaged graphulo jar unless filename is given) full path for
                graphulo jar
            filename = (Optional) name of config file to use; assumed to be of the form
                "py4j_path=/path/to/directory/py4j/share/py4j/py4j0.x.jar
                 accumulo_path=/path/to/directory/*
                 graphulo_path=/path/to/directory/graphulo-3.0.0.jar
                 jtds_path=/path/to/directory/jtds-1.3.1.jar
                 postgresql_path=/path/to/directory/postgresql-42.2.24.jar
                 mysql_path=/path/to/directory/mysql-connector-java-8.0.26.jar"
            die_on_exit = (Optional, default True) Boolean whether JVM should close upon exiting python
        Output:
            start_java() = JavaConnector.gateway
        """
        class_path = "."
        for path_name in valid_paths:
            if path_name in paths.keys() and path_name != "py4j_path":
                path = paths[path_name]
                JavaConnector.paths[path_name] = path
                class_path += ":" + path
            elif path_name != "py4j_path":
                path = _get_default_path(path_name, default_paths[path_name], file=filename, silent=True)
                JavaConnector.paths[path_name] = path
                class_path += ":" + path

        if "py4j_path" in paths.keys():
            py4j_path = paths["py4j_path"]
        else:
            py4j_path = _get_default_path("py4j_path", default_paths["py4j_path"], file=filename, silent=True)
        JavaConnector.paths["py4j_path"] = py4j_path

        port = py4j.java_gateway.launch_gateway(
            jarpath=py4j_path, classpath=class_path, die_on_exit=die_on_exit
        )

        gateway = py4j.java_gateway.JavaGateway(
            gateway_parameters=py4j.java_gateway.GatewayParameters(port=port),
            callback_server_parameters=py4j.java_gateway.CallbackServerParameters(
                port=0
            ),
        )

        print("JavaGateway started in Port " + str(port))

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
    """A DbServer instance collects the data needed to bind to a database instance.
    Attributes:
        instance_name = name of the database instance, e.g., the name of the Accumulo instance directory
        host = hostname associated with the database instance
        user = username associated with the database instance
        password = password associated with the database instance
        db_type = type of database (e.g., BigTableLike (Accumulo), sqlserver, mysql, pgres)
        gateway = py4j.java_gateway.JavaGateway object
    Note:
        - instance_name, host, user, and password are used as arguments of edu.mit.ll.d4m.db.cloud.D4mDbInfo
    """

    def __init__(
        self,
        instance_name: str,
        host: str,
        user: str,
        password: str,
        db_type: str,
        gateway: py4j.java_gateway.JavaGateway,
    ):
        """Construct DbServer instance from database connection information."""
        self.instance_name = instance_name
        self.host = host
        self.user = user
        self.password = password
        self.db_type = db_type
        self.gateway = gateway

    def ls(self) -> List[str]:
        """Print list of tables in DbServer instance."""
        if self.db_type in ["BigTableLike", "Accumulo"]:
            db_info = self.gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDbInfo(
                self.instance_name, self.host, self.user, self.password
            )
            table_value_str = db_info.getTableList().split(" ")
            table_value_str.pop()
        elif self.db_type == "sqlserver":
            sql_connection = db_sql_connect(self)
            query = sql_connection.prepareStatement("select * from sys.Tables")
            results = query.executeQuery()
            metadata = results.getMetaData()
            table_value_str = ""
            for j in [1, 7, 8, 9]:
                table_value_str += str(metadata.getColumnName(j)) + ","
            table_value_str += "\n"
            while results.next():
                for j in [1, 7, 8, 9]:
                    table_value_str += str(results.getString(j)) + ","
                table_value_str += "\n"
            sql_connection.close()
        elif self.db_type == "mysql":
            sql_connection = db_sql_connect(self)
            query = sql_connection.prepareStatement("show tables;")
            results = query.executeQuery()
            metadata = results.getMetaData()
            num_cols = metadata.getColumnCount()
            table_value_str = ""
            for j in range(num_cols):
                table_value_str += str(metadata.getColumnName(j)) + ","
            table_value_str += "\n"
            while results.next():
                for j in range(num_cols):
                    table_value_str += str(results.getString(j)) + ","
                table_value_str += "\n"
            sql_connection.close()
        elif self.db_type == "pgres":
            sql_connection = db_sql_connect(self)
            query = sql_connection.prepareStatement("select * from pg_catalog.pg_tables;")
            results = query.executeQuery()
            metadata = results.getMetaData()
            num_cols = metadata.getColumnCount()
            table_value_str = ""
            for j in range(num_cols):
                table_value_str += str(metadata.getColumnName(j)) + ","
            table_value_str += "\n"
            while results.next():
                for j in range(num_cols):
                    table_value_str += str(results.getString(j)) + ","
                table_value_str += "\n"
            sql_connection.close()
        else:
            table_value_str = ""

        return table_value_str

    def __getitem__(self, item: Union[str, Tuple[str, str]]) -> "DbTableLike":
        if isinstance(item, str):
            return get_index(self, item)
        else:
            return get_index(self, item[0], item[1])


class DbTableParent:
    def __init__(
        self,
        DB: DbServer,
        names: Union[str, Tuple[str, str]],
        security: str = "",
        num_limit: int = 0,
        num_row: int = 0,
        column_family: str = "",
        put_bytes: float = 5e5,
    ):
        self.DB = DB
        self.names = (names,) if isinstance(names, str) else names
        self.security = security
        self.num_limit = num_limit
        self.num_row = num_row
        self.column_family = column_family
        self.put_bytes = put_bytes
        self.iter_index = 0

        gateway = self.DB.gateway
        if DB.db_type in ["BigTableLike", "Accumulo"]:
            table_ops = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDbTableOperations

            self.table_ops = table_ops(
                self.DB.instance_name, self.DB.host, self.DB.user, self.DB.password
            )
        elif DB.db_type in ["sqlserver", "mysql", "pgres"]:
            assert len(names) == 1
            table_ops = gateway.jvm.edu.mit.ll.d4m.db.cloud.sql.D4mDbOperations  # Doesn't exist?
            self.table_ops = table_ops(DB.host)
        else:
            raise ValueError("Unsupported database type: " + str(DB.db_type))

        for table_name in self.names:
            if table_name not in self.DB.ls():
                print("Creating " + table_name + " in " + self.DB.instance_name + ".")
                self.table_ops.createTable(table_name)

        if DB.db_type in ["BigTableLike", "Accumulo"]:
            data_search = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDataSearch
            self.d4m_query = data_search(
                self.DB.instance_name,
                self.DB.host,
                self.names[0],
                self.DB.user,
                self.DB.password,
            )

            self.d4m_query.setCloudType(self.DB.db_type)
            self.d4m_query.setLimit(self.num_limit)
            self.d4m_query.reset()

        elif DB.db_type in ["sqlserver", "mysql", "pgres"]:
            sql_connection = db_sql_connect(DB)
            query = sql_connection.createStatement(gateway.jvm.java.sql.ResultSet.TYPE_SCROLL_SENSITIVE,
                                                   gateway.jvm.java.sql.ResultSet.CONCUR_READ_ONLY)
            self.d4m_query = query.executeQuery()

        else:
            pass  # Would have already thrown an error

    def nnz(self) -> int:
        table_list = self.DB.gateway.jvm.java.util.ArrayList()
        table_list.add(self.names[0])
        nnz_ = self.table_ops.getNumberOfEntries(table_list)
        return nnz_

    def set_limit(self, num_limit: int) -> None:
        self.num_limit = num_limit
        self.d4m_query.setLimit(num_limit)
        return None

    def do_matlab_query(self, row_query: str, col_query: str) -> None:
        self.d4m_query.reset()
        self.d4m_query.doMatlabQuery(
            row_query, col_query, self.column_family, self.security
        )
        return None

    def reset(self) -> None:
        self.d4m_query.setCloudType(self.DB.db_type)
        self.d4m_query.setLimit(self.num_limit)
        self.d4m_query.reset()
        self.do_matlab_query(":", ":")
        self.iter_index = 0
        return None

    def copy(self) -> "DbTableParent":
        return DbTableParent(
            self.DB,
            self.names,
            security=self.security,
            num_limit=self.num_limit,
            num_row=self.num_row,
            column_family=self.column_family,
            put_bytes=self.put_bytes,
        )

    def next(self) -> None:
        if self.has_next():
            self.d4m_query.next()
            self.iter_index += 1
        else:
            print("End of table reached.")
        return None

    def has_next(self) -> bool:
        return self.d4m_query.hasNext()

    def get_row_return_string(self):
        return D4M.util.from_db_string(self.d4m_query.getRowReturnString())

    def get_col_return_string(self):
        return D4M.util.from_db_string(self.d4m_query.getColumnReturnString())

    def get_val_return_string(self):
        return D4M.util.from_db_string(self.d4m_query.getValueReturnString())


class DbTable(DbTableParent):
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

    def __init__(
        self,
        DB: DbServer,
        name: str,
        security: str = "",
        num_limit: int = 0,
        num_row: int = 0,
        column_family: str = "",
        put_bytes: float = 5e5,
    ):
        super().__init__(
            DB,
            name,
            security=security,
            num_limit=num_limit,
            num_row=num_row,
            column_family=column_family,
            put_bytes=put_bytes,
        )

        self.name = name

    def copy(self):
        return DbTable(
            self.DB,
            self.name,
            security=self.security,
            num_limit=self.num_limit,
            num_row=self.num_row,
            column_family=self.column_family,
            put_bytes=self.put_bytes,
        )

    def put_triple(self, row: ArrayLike, col: ArrayLike, val: ArrayLike) -> None:
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
        gateway = self.DB.gateway

        new_row = D4M.util.num_to_str(row)
        new_col = D4M.util.num_to_str(col)
        new_val = D4M.util.num_to_str(val)

        # Optimize by selecting appropriate chunk size for insertion
        chunk_bytes = self.put_bytes
        num_triples = len(row)
        avg_byte_per_triple = (
            np.char.str_len(new_row).sum()
            + np.char.str_len(new_col).sum()
            + np.char.str_len(new_val).sum()
        ) / num_triples
        chunk_size = int(
            min(max(1, np.round(chunk_bytes / avg_byte_per_triple)), num_triples)
        )

        db_insert = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDbInsert
        insert_obj = db_insert(
            self.DB.instance_name,
            self.DB.host,
            self.name,
            self.DB.user,
            self.DB.password,
        )

        for chunk_start in np.arange(0, num_triples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_triples)
            new_row_chunk = D4M.util.to_db_string(new_row[chunk_start:chunk_end])
            new_col_chunk = D4M.util.to_db_string(new_col[chunk_start:chunk_end])
            new_val_chunk = D4M.util.to_db_string(new_val[chunk_start:chunk_end])

            insert_obj.doProcessing(
                new_row_chunk,
                new_col_chunk,
                new_val_chunk,
                self.column_family,
                self.security,
            )
        return None

    def __getitem__(
        self, query: Union[None, Tuple[ArrayLike, ArrayLike]]
    ) -> D4M.assoc.Assoc:
        if query is not None:
            return get_index(self, *query)
        else:
            return get_index(self)


class DbTablePair(DbTableParent):
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

    def __init__(
        self,
        DB: DbServer,
        name_1: str,
        name_2: str,
        security: str = "",
        num_limit: int = 0,
        num_row: int = 0,
        column_family: str = "",
        put_bytes: float = 5e5,
    ):
        super().__init__(
            DB,
            (name_1, name_2),
            security=security,
            num_limit=num_limit,
            num_row=num_row,
            column_family=column_family,
            put_bytes=put_bytes,
        )

        self.name_1 = name_1
        self.name_2 = name_2

    def copy(self):
        return DbTablePair(
            self.DB,
            self.name_1,
            self.name_2,
            security=self.security,
            num_limit=self.num_limit,
            num_row=self.num_row,
            column_family=self.column_family,
            put_bytes=self.put_bytes,
        )

    def put_triple(self, row: ArrayLike, col: ArrayLike, val: ArrayLike) -> None:
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
        gateway = self.DB.gateway

        new_row = D4M.util.num_to_str(row)
        new_col = D4M.util.num_to_str(col)
        new_val = D4M.util.num_to_str(val)

        # Optimize by selecting appropriate chunk size for insertion
        chunk_bytes = self.put_bytes
        num_triples = len(row)
        avg_byte_per_triple = (
            np.char.str_len(new_row).sum()
            + np.char.str_len(new_col).sum()
            + np.char.str_len(new_val).sum()
        ) / num_triples
        chunk_size = int(
            min(max(1, np.round(chunk_bytes / avg_byte_per_triple)), num_triples)
        )

        db_insert = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDbInsert
        insert_obj = db_insert(
            self.DB.instance_name,
            self.DB.host,
            self.name_1,
            self.DB.user,
            self.DB.password,
        )
        insert_objT = db_insert(
            self.DB.instance_name,
            self.DB.host,
            self.name_2,
            self.DB.user,
            self.DB.password,
        )

        for chunk_start in np.arange(0, num_triples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_triples)
            new_row_chunk = D4M.util.to_db_string(new_row[chunk_start:chunk_end])
            new_col_chunk = D4M.util.to_db_string(new_col[chunk_start:chunk_end])
            new_val_chunk = D4M.util.to_db_string(new_val[chunk_start:chunk_end])

            insert_obj.doProcessing(
                new_row_chunk,
                new_col_chunk,
                new_val_chunk,
                self.column_family,
                self.security,
            )
            insert_objT.doProcessing(
                new_col_chunk,
                new_row_chunk,
                new_val_chunk,
                self.column_family,
                self.security,
            )
        return None

    def __getitem__(
        self, query: Union[None, Tuple[ArrayLike, ArrayLike]]
    ) -> D4M.assoc.Assoc:
        if query is not None:
            return get_index(self, *query)
        else:
            return get_index(self)


DbTableLike = Union[DbTableParent, DbTable, DbTablePair]


default_tables = ["accumulo.metadata", "accumulo.replication", "accumulo.root", "trace"]


def db_sql_connect(DB: DbServer):
    props = DB.gateway.java.util.Properties
    user = DB.user
    sep_location = user.find("\\")
    if sep_location > -1:
        props.put('domain', user[:sep_location])
        user = user[sep_location + 1:]

    if DB.db_type == 'sqlserver':
        driver = DB.gateway.net.sourceforge.jtds.jdbc.Driver
        connection_str = "jdbc:jtds:sqlserver://" + DB.host + "/" + DB.instance_name \
                         + ";user=" + user + ";password=" + DB.password
    elif DB.db_type == 'pgres':
        driver = DB.gateway.org.postgresql.Driver
        connection_str = "jdbc:postgresql://" + DB.host + "/" + DB.instance_name \
                         + "?user=" + user + "&password=" + DB.password
    elif DB.db_type == 'mysql':
        driver = DB.gateway.org.gjt.mm.mysql.Driver
        if len(DB.password) == 0:
            connection_str = "jdbc:mysql://" + DB.host + "/" + DB.instance_name + "?user=" + user
        else:
            connection_str = "jdbc:mysql://" + DB.host + "/" + DB.instance_name \
                             + "?user=" + user + "&password=" + DB.password
    else:
        return None

    connection = driver.connect(connection_str, props)
    return connection


def dbsetup_sql(
        instance: str,
        db_type: str,
        hostname: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        force_restart: bool = False,
        die_on_exit: bool = True,
        **paths
):
    if username is None:
        username = input("Username:")
    if password is None:
        password = getpass.getpass(prompt="Password:")

    if JavaConnector.jvm_gateway is None or force_restart:
        gateway = JavaConnector.start_java(
            die_on_exit=die_on_exit,
            paths=paths
        )
    else:
        gateway = JavaConnector.jvm_gateway

    return DbServer(instance, hostname, username, password, db_type, gateway)


def dbsetup_LLgrid(
    instance: str,
    config: Optional[str] = None,
    filename: Optional[str] = None,
    force_restart: bool = False,
    die_on_exit: bool = True,
    **paths
) -> DbServer:
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
                        "instance=*actual instance name*
                         hostname=*actual hostname*
                         username=*actual username*
                         password=*actual password*"
        py4j_path, accumulo_path, graphulo_path, force_restart, die_on_exit = see JavaConnector.start_java()
    Output:
        DB = DbServer containing the connection information to Accumulo instance
    Examples:
        dbsetup('class-db49')
        dbsetup('class-db50')
    """
    if config is None:
        config = _get_default_path(
            "config", "/home/gridsan/tools/groups/", file=filename, silent=True
        )

    if os.path.isdir(config):
        dbdir = config + "/databases/" + instance
        with open(dbdir + "/accumulo_user_password.txt", "r") as f:
            pword = f.read()
        with open(dbdir + "/dnsname", "r") as f:
            hostname = f.read()
            hostname = hostname.replace("\n", "") + ":2181"
            username = "AccumuloUser"
    elif os.path.isfile(config):
        with open(config, "r") as f:
            conf = [line.split("=") for line in f.readlines()]
            conf = {line[0]: line[1] for line in conf}
        instance = conf["instance"]
        hostname = conf["hostname"]
        username = conf["username"]
        pword = conf["password"]
    else:
        raise ValueError(
            "'config' must either be a config file or a directory containing a config file."
            + "Supplied 'config': "
            + str(config)
        )

    if JavaConnector.jvm_gateway is None or force_restart:
        gateway = JavaConnector.start_java(
            die_on_exit=die_on_exit,
            paths=paths
        )
    else:
        gateway = JavaConnector.jvm_gateway

    return DbServer(instance, hostname, username, pword, "BigTableLike", gateway)


def _get_index_single(DB: DbServer, table_name: str, **table_options) -> DbTable:
    """Create DbTable object containing binding information for tableName in DB."""
    return DbTable(DB, table_name, **table_options)


def _get_index_pair(
    DB: DbServer, table_name_1: str, table_name_2: str, **table_options
) -> DbTablePair:
    """Create DbTablePair object containing binding information for tableName1 and tableName2 in DB."""
    return DbTablePair(DB, table_name_1, table_name_2, **table_options)


_colon_equivalents = [
    ":",
    slice(None, None),
    slice(0, None),
    slice(None, None, 1),
    slice(0, None, 1),
]


def _is_startswith(query):
    name = getattr(query, "__name__", "Unknown")
    return name.startswith("startswith")


def _is_valid_query(query):
    if _is_startswith(query):
        return True
    elif callable(query) or (
        isinstance(query, slice) and query not in _colon_equivalents
    ):
        return False
    else:
        query = D4M.util.sanitize(query)
        if query.dtype == int or (
            query.dtype == str
            and ":" in query
            and (len(query) != 1 and len(query) != 3)
        ):
            return False
        else:
            return True


def _is_colon(object_):
    return isinstance(object_, str) and object_ == ":"


def _get_index_assoc(
    table: DbTableLike, row_query: Selectable, col_query: Selectable
) -> Union[D4M.assoc.Assoc, None]:
    """Create Assoc object from the sub-array of table with queried row and column indices/keys."""
    switch_keys = False

    row_query = ":" if row_query in _colon_equivalents else row_query
    if row_query not in _colon_equivalents and D4M.util.can_sanitize(row_query):
        row_query = D4M.util.sanitize(row_query)
    col_query = ":" if col_query in _colon_equivalents else col_query
    if col_query not in _colon_equivalents and D4M.util.can_sanitize(col_query):
        col_query = D4M.util.sanitize(col_query)

    if not _is_colon(col_query):
        if isinstance(table, DbTable):
            if table.num_limit > 0:
                raise ValueError("Column queries are not supported for DbTable iterators.")
            else:
                table.d4m_query.setTableName(table.name)
        else:
            assert isinstance(table, DbTablePair)
            if _is_colon(row_query):
                table.d4m_query.setTableName(table.name_2)
                row_query, col_query = col_query, row_query
                switch_keys = True
            elif table.num_limit > 0:
                raise ValueError("DbTablePair iterators do not support both row and column queries being nontrivial.")
            else:
                table.d4m_query.setTableName(table.name_1)
    else:
        table_name = table.name if isinstance(table, DbTable) else table.name_1
        table.d4m_query.setTableName(table_name)

    table.d4m_query.setCloudType(table.DB.db_type)
    table.d4m_query.setLimit(table.num_limit)

    if not _is_valid_query(row_query) or not _is_valid_query(col_query):
        warnings.warn("Row and column queries must either be string lists, startswith functions, or "
                      "full slices (e.g., :).")
    else:
        if not _is_colon(row_query):
            row_query = D4M.util.to_db_string(row_query)
        table.d4m_query.reset()
        table.d4m_query.doMatlabQuery(row_query, ":", table.column_family, table.security)
        row = table.get_row_return_string()
        col = table.get_col_return_string()
        val = table.get_val_return_string()

        table_assoc = D4M.assoc.Assoc(row, col, val)

        if not _is_colon(col_query):
            table_assoc = table_assoc[:, col_query]

        if switch_keys:
            table_assoc.transpose(copy=False)

        return table_assoc


def _get_index_from_iter(table: DbTableLike) -> D4M.assoc.Assoc:
    """Query table as iterator if table.num_limit > 0 and return associative array consisting of next batch of triples,
    otherwise return associative array consisting of all triples."""
    if table.num_limit == 0:
        return get_index(table, ":", ":")

    table_name = table.d4m_query.getTableName()

    if table.has_next():
        table.next()

        row = table.get_row_return_string()
        col = table.get_col_return_string()
        val = table.get_val_return_string()

        if isinstance(table, DbTablePair) and table_name == table.name_2:
            row, col = col, row
    else:
        row, col, val = [], [], []

    return D4M.assoc.Assoc(row, col, val)


def get_index(*arg, **table_options) -> Union[DbTableLike, D4M.assoc.Assoc]:
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
        output = _get_index_single(DB, table_name, **table_options)
    elif (
        len(arg) == 3
        and isinstance(arg[0], DbServer)
        and isinstance(arg[1], str)
        and isinstance(arg[2], str)
    ):
        DB, table_name_1, table_name_2 = arg
        output = _get_index_pair(DB, table_name_1, table_name_2, **table_options)
    elif len(arg) == 3 and (
        isinstance(arg[0], DbTable) or isinstance(arg[0], DbTablePair)
    ):
        # Assume arg[1] and arg[2] can be interpreted as row and column queries, respectively
        table, row_query, column_query = arg
        output = _get_index_assoc(table, row_query, column_query)
    elif len(arg) == 1 and (
        isinstance(arg[0], DbTable) or isinstance(arg[0], DbTablePair)
    ):
        table = arg[0]
        output = _get_index_from_iter(table)
    else:
        raise ValueError(
            "Improper argument supplied. Argument must be of the form 'DB, table_name', "
            "'DB, table_name_1, table_name_2', 'table, row_query, col_query', or 'table'."
        )
    return output


valid_confirm = ["y", "yes", "Y", "Yes"]  # Enumerate user inputs that 'confirm'


def _delete_table_single(table: DbTable, force: bool = False) -> None:
    """Delete the table with name table.name in table.DB instance."""
    if table.name in table.DB.ls():
        if table.name in default_tables:
            print(table.name + " is a default table and cannot be deleted.")
        else:
            if force:
                confirm = "yes"
            else:
                confirm = input(
                    "Confirm deletion of "
                    + table.name
                    + " in "
                    + table.DB.instance_name
                    + "."
                )
            if confirm in valid_confirm:
                table.table_ops.deleteTable(table.name)
                print("Deleted " + table.name + " from " + table.DB.instance_name + ".")
    else:
        print(table.name + " was not found in " + table.DB.instance_name + ".")
    return None


def _delete_table_pair(table_pair: DbTablePair, force: bool = False) -> None:
    """Delete the tables with names table.name1 and table.name2 in table.DB instance."""
    present_tables = table_pair.DB.ls()
    table_pair_names = [table_pair.name_1, table_pair.name_2]
    is_present_list = [
        table_pair.name_1 in present_tables,
        table_pair.name_2 in present_tables,
    ]
    is_default_list = [
        table_pair.name_1 in default_tables,
        table_pair.name_2 in default_tables,
    ]

    for table_index in range(2):
        is_present = is_present_list[table_index]
        is_default = is_default_list[table_index]
        table_name = table_pair_names[table_index]
        if is_present and not is_default:
            if force:
                confirm = "yes"
            else:
                confirm = input(
                    "Confirm deletion of "
                    + table_name
                    + " in "
                    + table_pair.DB.instance_name
                    + "."
                )
            if confirm in valid_confirm:
                table_pair.table_ops.deleteTable(table_name)
                print(
                    "Deleted "
                    + table_name
                    + " from "
                    + table_pair.DB.instance_name
                    + "."
                )
        elif is_present and is_default:
            print(table_name + " is a default table and cannot be deleted.")
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
        confirm = "yes"
    else:
        confirm = input(
            "Confirm deletion of all non-default tables in " + DB.instance_name + "."
        )
    if confirm in valid_confirm:
        for table_name in present_tables:
            temp_table = get_index(DB, table_name)
            if table_name not in default_tables:
                delete_table(temp_table, force=True)
    return None


def get_iterator(table: DbTableLike, num_limit: int) -> DbTableLike:
    """Query iterator functionality."""
    table_iter = table.copy()
    table_iter.set_limit(num_limit)
    table_iter.reset()
    return table_iter


def put_triple(
    table: DbTableLike, row: ArrayLike, col: ArrayLike, val: ArrayLike
) -> None:
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
    table.put_triple(row, col, val)
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
    return table.nnz()


def add_splits(
    table: DbTableLike, split_string: str, split_stringT: Optional[str] = None
) -> None:
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
