from py4j.java_gateway import JavaGateway
from py4j.java_gateway import launch_gateway
from py4j.java_gateway import GatewayParameters
from py4j.java_gateway import CallbackServerParameters
import numpy as np

import os
import sys

# Facilitate jars being referenced
from pkg_resources import resource_filename

from D4M import assoc as As

# Support raw_input between Python 2.x & Python 3.x
if sys.version[0] == 3:
    raw_input = input

class JavaConnector:
    """
        Handles the starting of the JVM and exposing the JVM to the relevant JARs of Graphulo and Accumulo.
    """

    jvm_port = None
    jvm_gateway = None

    @staticmethod
    def start_java(py4j_path=None, accumulo_path=None, graphulo_path=None, dieonexit=None):
        """
            Starts JVM along with a Py4J JavaGateway instance, with classpath exposing
            Accumulo and Graphulo to the JVM.

            Inputs:
                py4j_path = full path for py4j jar
                accumulo_path = full path of directory containing Accumulo jars
                graphulo_path = full path for graphulo jar
                dieonexit = option to have JVM close upon exiting of python

            Note:
                - By default, uses packaged JARs
        """

        if py4j_path is None:
            py4j_path = resource_filename(__name__, '/jars/py4j/share/py4j/py4j.0.1.jar')
        if accumulo_path is None:
            accumulo_path = resource_filename(__name__, '/jars/libext/*')
        if graphulo_path is None:
            graphulo_path = resource_filename(__name__, '/jars/lib/graphulo-3.0.0.jar')

        cp = '.:'+accumulo_path+':'+graphulo_path

        if dieonexit is None:
            dieonexit = True

        port = launch_gateway(jarpath=py4j_path, classpath=cp, die_on_exit=dieonexit)

        gateway = JavaGateway(
            gateway_parameters=GatewayParameters(port=port),
            callback_server_parameters=CallbackServerParameters(port=0))

        print('JavaGateway started in Port '+str(port))
        
        JavaConnector.jvm_port = port
        JavaConnector.jvm_gateway = gateway

        return gateway

    @staticmethod
    def getport():
        """ Gets the port through which the JavaGateway is communicating. """
        return JavaConnector.jvm_port
    
    @staticmethod
    def getgateway():
        """ Gets the JavaGateway instance. """
        return JavaConnector.jvm_gateway


class Dbserver:
    """ Class containing the database connection information needed to bind to Accumulo tables. """

    def __init__(self, instanceName, host, user, password, dbType, gateway):
        """ Constructor from db connection information. """
        self.instanceName = instanceName
        self.host = host
        self.user = user
        self.password = password
        self.dbType = dbType
        self.gateway = gateway

    def ls(self):
        """ Provides a list of the tables in self.instanceName. """
        gateway = self.gateway
        dbInfo = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDbInfo(self.instanceName,
                                                               self.host, self.user, self.password)
        tables = dbInfo.getTableList().split(' ')
        tables.pop()
        return tables


class Dbtable:
    """ Binding information to existing table. """

    def __init__(self, DB, name, security, numLimit, numRow,
                 columnfamily, putBytes, d4mQuery, tableOps):
        self.DB = DB
        self.name = name
        self.security = security
        self.numLimit = numLimit
        self.numRow = numRow
        self.columnfamily = columnfamily
        self.putBytes = putBytes
        self.d4mQuery = d4mQuery
        self.tableOps = tableOps


class Dbtablepair:
    """ Binding information to existing pair of tables. """

    def __init__(self, DB, name1, name2, security, numLimit,
                 numRow, columnfamily, putBytes, d4mQuery, tableOps):
        self.DB = DB
        self.name1 = name1
        self.name2 = name2
        self.security = security
        self.numLimit = numLimit
        self.numRow = numRow
        self.columnfamily = columnfamily
        self.putBytes = putBytes
        self.d4mQuery = d4mQuery
        self.tableOps = tableOps


if sys.version[0] == 3:
    raw_input = input


def dbsetup(instance, config=None, py4j_path=None, accumulo_path=None, graphulo_path=None, dieonexit=None,
            forcerestart=None):
    """
    Sets up a DB connection, starting JVM if not already started.
        Usage:
            dbsetup('instance_name')
            dbsetup('instance_name','config_location')
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
                             passowrd = *actual password*"
            py4j_path,
             accumulo_path,
             graphulo_path,
             dieonexit = options for start_java()
        Output:
            DB = Dbserver, containing the connection information to Accumulo instance
        Examples:
            dbsetup('class-db49')
            dbsetup('class-db50')
    """
    
    # Set config to MIT Supercloud default location if None given
    if config is None:
        config = '/home/gridsan/tools/groups/'

    if os.path.isdir(config) or os.path.isfile(config):
        if os.path.isdir(config):
            dbdir = config + '/databases/' + instance
            with open(dbdir + '/accumulo_user_password.txt', 'r') as f:
                pword = f.read()
            with open(dbdir + '/dnsname', 'r') as f:
                hostname = f.read()
                hostname = hostname.replace('\n', '') + ':2181'
                username = "AccumuloUser"
        else:
            with open(config, 'r') as f:
                conf = [line.split('=') for line in f.readlines()]
                conf = {line[0]: line[1] for line in conf}
            instance = conf['instance']
            hostname = conf['hostname']
            username = conf['username']
            pword = conf['password']

        if JavaConnector.jvm_gateway is None or forcerestart:
            gateway = JavaConnector.start_java(py4j_path=py4j_path, 
                                               accumulo_path=accumulo_path, 
                                               graphulo_path=graphulo_path, 
                                               dieonexit=dieonexit)
        else:
            gateway = JavaConnector.jvm_gateway

        DB = Dbserver(instance, hostname, username, pword, "BigTableLike", gateway)
    else:
        print("Provide either a directory or config file.")
        DB = None

    return DB


def getindexsingle(DB, tableName):
    """
    Create DBtable object containing binding information for tableName in DB.
        Usage:
            getindexsingle(DB,tableName)
        Input:
            DB = DBserver - server information for accumulo instance
            tableName = name of existing table in accumulo instance to bind to or of table to create in instance
        Output:
            getindexsingle(DB,tablename) = DBtable with binding information for tableName, if it exists,
                                            otherwise creates a table in DB with default settings, and
                                            binds to that
    """
    gateway = DB.gateway
    opsObj = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDbTableOperations(DB.instanceName,
                                                                      DB.host, DB.user, DB.password)

    if tableName not in DB.ls():
        print("Creating " + tableName + " in " + DB.instanceName + ".")
        opsObj.createTable(tableName)

    queryObj = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDataSearch(DB.instanceName, DB.host,
                                                                 tableName, DB.user,
                                                                 DB.password)

    dbtab = Dbtable(DB, tableName, '', 0, 0, '', 5e5, queryObj, opsObj)
    return dbtab


def getindexpair(DB, tableName1, tableName2):
    """
    Create Dbtablepair object containing binding information for tableName1 and tableName2 in DB.
        Usage:
            getindexpair(DB,tableName1,tablename2)
        Input:
            DB = DBserver - server information for accumulo instance
            tableName1 = name of existing table in accumulo instance to bind to or of table to create in instance
            tableName2 = name of existing table in accumulo instance to bind to or of table to create in instance
        Output:
            getindexpair(DB,tablename1,tablename2) = DBtable with binding information for tableName1 and tablename2,
                                                    if they exist, otherwise creates necessary tables in DB
                                                    with default settings, and binds to those
    """
    gateway = DB.gateway
    opsObj = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDbTableOperations(DB.instanceName,
                                                                      DB.host, DB.user, DB.password)
    for tableName in [tableName1, tableName2]:
        if tableName not in DB.ls():
            print("Creating " + tableName + " in " + DB.instanceName + ".")
            opsObj.createTable(tableName)

    queryObj = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDataSearch(DB.instanceName, DB.host,
                                                                 tableName, DB.user,
                                                                 DB.password)

    dbtabpair = Dbtablepair(DB, tableName1, tableName2, '', 0, 0, '', 5e5, queryObj, opsObj)
    return dbtabpair


def deletetablesingle(table):
    """
    Delete the table with name table.name in table.DB instance.
    WARNING: Deletes the *actual* table, not the binding object.
    """
    if table.name in table.DB.ls():
        confirm = raw_input("Confirm deletion of " + table.name + " in " + table.DB.instanceName + ".")
        if confirm in ['y', 'yes', 'Yes', 'Y']:
            table.tableOps.deleteTable(table.name)
            print("Deleted " + table.name + " from " + table.DB.instanceName + ".")
    else:
        print(table.name + " was not found in " + table.DB.instanceName + ".")
    return None


def deletetablepair(tablepair):
    """
    Delete the tables with names table.name1 and table.name2 in table.DB instance.
    WARNING: Deletes the *actual* tables, not the binding object.

    Note: option provided to delete one of the two tables but not the other.
    """
    validconfirm = ['y', 'yes', 'Y', 'Yes']  # Enumerate user inputs that 'confirm'

    # Go through the various combinations of whether the individual tables are actually present in the database
    if tablepair.name1 in tablepair.DB.ls() and tablepair.name2 not in tablepair.DB.ls():
        print(tablepair.name2 + " was not found in " + tablepair.DB.instanceName + ".")
        confirm = raw_input("Confirm deletion of " + tablepair.name1 + " in " + tablepair.DB.instanceName + ".")
        if confirm in validconfirm:
            tablepair.tableOps.deleteTable(tablepair.name1)
            print("Deleted " + tablepair.name1 + " from " + tablepair.DB.instanceName + ".")
    elif tablepair.name1 not in tablepair.DB.ls() and tablepair.name2 in tablepair.DB.ls():
        print(tablepair.name1 + " was not found in " + tablepair.DB.instanceName + ".")
        confirm = raw_input("Confirm deletion of " + tablepair.name2 + " in " + tablepair.DB.instanceName + ".")
        if confirm in validconfirm:
            tablepair.tableOps.deleteTable(tablepair.name2)
            print("Deleted " + tablepair.name2 + " from " + tablepair.DB.instanceName + ".")
    elif tablepair.name1 in tablepair.DB.ls() and tablepair.name2 in tablepair.DB.ls():
        confirm = raw_input("Confirm deletion of " + tablepair.name1 + " and " + tablepair.name2
                            + " in " + tablepair.DB.instanceName + ".")
        if confirm in ['y', 'yes', 'Yes', 'Y']:
            tablepair.tableOps.deleteTable(tablepair.name1)
            tablepair.tableOps.deleteTable(tablepair.name2)
            print("Deleted " + tablepair.name1 + " and " + tablepair.name2 + " in " + tablepair.DB.instanceName + ".")
    else:
        print(tablepair.name1 + " and " + tablepair.name2 + " were not found in " + tablepair.DB.instanceName + ".")
    return None


def deletetable(table):
    """ Wrapper for deletetablesingle and deletetablepair. """
    if isinstance(table, Dbtablepair):
        deletetablepair(table)
    else:
        deletetablesingle(table)

    return None


def todbstring(i):
    """ Convert input i (either a delimiter-separated string or an iterable) to Accumulo-friendly string. """
    if isinstance(i, str):
        i = i.replace(i[-1], '\n')
    if hasattr(i, '__iter__'):
        i = '\n'.join(i) + '\n'
    return i


def getindexassoc(table, i, j):
    """
    Create Assoc object from the sub-array of table with row and column indices from i and j, resp.
    Note: currently does not support slices, startswith, or contains.
    """
    # TODO: support arbitrary slicing, starswith, and contains

    if i != ":":
        i = todbstring(i)
    if j != ":":
        j = todbstring(j)

    table.d4mQuery.setCloudType(table.DB.dbType)
    table.d4mQuery.setLimit(table.numLimit)
    table.d4mQuery.reset()

    if i != ":" or j == ":" or isinstance(table, Dbtable):
        if isinstance(table, Dbtablepair):
            table.d4mQuery.setTableName(table.name1)

        table.d4mQuery.doMatlabQuery(i, j, table.columnfamily, table.security)

        r = table.d4mQuery.getRowReturnString()
        c = table.d4mQuery.getColumnReturnString()

    else:
        table.d4mQuery.setTableName(table.name2)

        table.d4mQuery.doMatlabQuery(i, j, table.columnfamily, table.security)

        c = table.d4mQuery.getRowReturnString()
        r = table.d4mQuery.getColumnReturnString()

    v = table.d4mQuery.getValueReturnString()

    A = As.Assoc(r, c, v)
    return A


def getiterator(table, nelements):
    """ Query iterator functionality. """
    gateway = table.DB.gateway()
    ops = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDbTableOperations
    opsObj = ops(table.DB.instanceName, table.DB.host, table.DB.user, table.DB.password)

    d4mQuery = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDataSearch
    if isinstance(table, Dbtablepair):
        queryObj = d4mQuery(table.DB.instanceName, table.DB.host,
                            table.name1, table.name2, table.DB.user, table.DB.password)
        Ti = Dbtablepair(table.DB, table.name1, table.name2, table.security,
                         nelements, table.numRow, table.columnfamily, table.putBytes, queryObj, opsObj)
    else:
        queryObj = d4mQuery(table.DB.instanceName, table.DB.host,
                            table.name, table.DB.user, table.DB.password)
        Ti = Dbtable(table.DB, table.name, table.security,
                     nelements, table.numRow, table.columnfamily, table.putBytes, queryObj, opsObj)

    return Ti


def getindexfromiter(table):
    """ Query from table using ambient iterator, outputing an Assoc object. """
    table.d4mQuery.next()

    tablename = table.d4mQuery.getTableName()

    if isinstance(table, Dbtablepair) and tablename == table.name2:
        c = table.d4mQuery.getRowReturnString()
        r = table.d4mQuery.getColumnReturnString()
    else:
        r = table.d4mQuery.getRowReturnString()
        c = table.d4mQuery.getColumnReturnString()

    v = table.d4mQuery.getValueReturnString()

    A = As.Assoc(r, c, v)
    return A


def getindex(*arg):
    """ Wrapper for getindexsingle, getindexpair, getindexassoc, and getindexfromiter. """
    if len(arg) == 2:
        DB, tablename = arg
        output = getindexsingle(DB, tablename)
    elif len(arg) == 3 and isinstance(arg[0], Dbserver):
        DB, tablename1, tablename2 = arg
        output = getindexpair(DB, tablename1, tablename2)
    elif len(arg) == 3:
        table, i, j = arg
        output = getindexassoc(table, i, j)
    elif len(arg) == 1:
        table = arg[0]
        output = getindexfromiter(table)
    else:
        print("Argument must be one of 'DB,tablename', 'DB,tablename1,tablename2', 'table,i,j', or 'table'.")
        output = None

    return output


def puttriple(table, r, c, v):
    """
    Insert the triples (r(i),c(i),v(i)) into table.
        Usage:
            putTriple(table,r,c,v)
        Inputs:
            table = DBtable or Dbtablepair instance
            r = string of (delimiter separated) values (delimiter is last character)
                or list of values of length n
            c = string of (delimiter separated) values (delimiter is last character)
                or list of values of length n
            v = string of (delimiter separated) values (delimiter is last character)
                or list of values of length n
        Notes:
            - If table is a Dbtablepair, then table.name2 is assumed to be the transpose of
                table.name1 for the purposes or insertion (e.g. transposed triples are put
                into table.nam2)
            - Accumulo tables record duplicate triples.
    """
    gateway = table.DB.gateway

    r = As.sanitize(r)
    c = As.sanitize(c)
    v = As.sanitize(v)

    # Optimize by selecting appropriate chunksize for insertion
    chunkBytes = table.putBytes
    numTriples = int(np.size(r))
    avgBytePerTriple = (np.vectorize(len)(r).sum() + np.vectorize(len)(c).sum()
                        + np.vectorize(len)(v).sum()) / numTriples
    chunkSize = int(min(max(1, np.round(chunkBytes / avgBytePerTriple)), numTriples))

    dbInsert = gateway.jvm.edu.mit.ll.d4m.db.cloud.D4mDbInsert
    if isinstance(table, Dbtablepair):
        insertObj = dbInsert(table.DB.instanceName, table.DB.host, table.name1, table.DB.user, table.DB.password)
        insertObjT = dbInsert(table.DB.instanceName, table.DB.host, table.name2, table.DB.user, table.DB.password)
    else:
        insertObj = dbInsert(table.DB.instanceName, table.DB.host, table.name, table.DB.user, table.DB.password)

    for i in np.arange(0, numTriples, chunkSize):
        iNext = min(i + chunkSize, numTriples)
        rr = todbstring(r[i:iNext])
        cc = todbstring(c[i:iNext])
        vv = todbstring(v[i:iNext])

        insertObj.doProcessing(rr, cc, vv, table.columnfamily, table.security)

        if isinstance(table, Dbtablepair):
            insertObjT.doProcessing(cc, rr, vv, table.columnfamily, table.security)

    return None


def put(table, A):
    """ Put Assoc object A into table. """
    r, c, v = A.find()
    puttriple(table, r, c, v)
    return None


def nnz(table):
    """
    Returns the number of non-zero triples stored in table.
    Note: returns count for all triples found, including duplicates.
    """
    if isinstance(table, Dbtablepair):
        tname = table.name1
    else:
        tname = table.name

    gateway = table.DB.gateway
    tablelist = gateway.jvm.java.util.ArrayList()
    tablelist.add(tname)

    nnz = table.tableOps.getNumberOfEntries(tablelist)
    return nnz
