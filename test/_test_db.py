# TODO: Figure out how to test locally
import pytest
from py4j.java_gateway import JavaGateway
from typing import Tuple

import D4M.assoc
import D4M.util as util
import D4M.db


@pytest.fixture(scope="session")
def test_instance():
    return "class-db03"


@pytest.fixture(scope="session")
def config_filename():
    return "test_db_config.txt"


@pytest.fixture(scope="session")
def DB(test_instance):
    DB = D4M.db.dbsetup(test_instance, die_on_exit=False)
    yield DB


def _get_new_table_name(DB: D4M.db.DbServer) -> str:
    """Create table name which isn't present in the given DbServer."""
    table_list = DB.ls()
    index = 0
    new_table_name = "test_table_"
    while new_table_name + str(index) in table_list:
        index += 1
    return new_table_name + str(index)


def _get_new_tablepair_name(DB: D4M.db.DbServer) -> Tuple[str, str]:
    """Create table pair names which aren't present in the given DbServer."""
    table_list = DB.ls()
    index = 0
    new_table_name = "test_table_"
    while (
        new_table_name + str(index) in table_list
        or new_table_name + str(index) + "T" in table_list
    ):
        index += 1
    return new_table_name + str(index), new_table_name + str(index) + "T"


# D4M.assoc.Assoc(test_row, test_col, test_val).printfull():
#           2   4   6   8   A      ABCD   B   C   D
# 1        12  14      18  1A                1C  1D
# 3            34  36  38         3ABCD  3B      3D
# 5        52  54  56  58                5B  5C
# 7        72      76  78  7A            7B
# a        a2  a4  a6  a8  aA                aC  aD
# abcd  abcd2                  abcdABCD
# b        b2      b6      bA            bB      bD
# c        c2      c6                        cC
# d                                              dD


@pytest.fixture(scope="function")
def row():
    return "1,1,1,1,1,1,3,3,3,3,3,3,5,5,5,5,5,5,7,7,7,7,7,a,a,a,a,a,a,a,abcd,abcd,b,b,b,b,b,c,c,c,d,"


@pytest.fixture(scope="function")
def col():
    return "2,4,8,A,C,D,4,6,8,ABCD,B,D,2,4,6,8,B,C,2,6,8,A,B,2,4,6,8,A,C,D,2,ABCD,2,6,A,B,D,2,6,C,D,"


@pytest.fixture(scope="function")
def val():
    return (
        "12,14,18,1A,1C,1D,34,36,38,3ABCD,3B,3D,52,54,56,58,5B,5C,72,76,78,7A,7B,a2,a4,a6,a8,"
        "aA,aC,aD,abcd2,abcdABCD,b2,b6,bA,bB,bD,c2,c6,cC,dD,"
    )


@pytest.fixture(scope="function")
def table(DB, row, col, val):
    table_name = _get_new_table_name(DB)
    table = D4M.db.get_index(DB, table_name)

    D4M.db.put_triple(table, row, col, val)

    yield table
    D4M.db.delete_table(table, force=True)


@pytest.fixture(scope="function")
def tablepair(DB, row, col, val):
    tablepair_name, tablepair_nameT = _get_new_tablepair_name(DB)
    tablepair = D4M.db.get_index(DB, tablepair_name, tablepair_nameT)

    D4M.db.put_triple(tablepair, row, col, val)

    yield tablepair
    D4M.db.delete_table(tablepair, force=True)


def test_start_java_gateway(config_filename):
    gateway = D4M.db.JavaConnector.start_java(filename=config_filename)
    assert isinstance(gateway, JavaGateway)


def test_start_java_jvm(config_filename):
    gateway = D4M.db.JavaConnector.start_java(filename=config_filename)
    java_func = gateway.jvm.java.lang.Math.pow
    assert java_func(2.0, 3.0) == 8.0


def test_dbsetup(DB, test_instance):
    assert isinstance(DB, D4M.db.DbServer)
    assert DB.instance_name == test_instance
    assert isinstance(DB.host, str)
    assert isinstance(DB.user, str)
    assert isinstance(DB.password, str)
    assert isinstance(DB.db_type, str)
    assert isinstance(DB.gateway, JavaGateway)


def test_ls(DB):
    table_list = DB.ls()
    assert isinstance(table_list, list)
    table_types = set([type(item) for item in table_list])
    assert table_types == {str}
    assert set(D4M.db.default_tables) <= set(table_list)


def test_index_and_delete_single(DB):
    new_table_name = _get_new_table_name(DB)
    assert new_table_name not in DB.ls()

    new_table = D4M.db.get_index(DB, new_table_name)
    assert isinstance(new_table, D4M.db.DbTable)
    assert new_table.name == new_table_name
    assert new_table.name in DB.ls()

    D4M.db.delete_table(new_table, force=True)
    assert new_table.name not in DB.ls()


def test_index_and_delete_single_getitem(DB):
    new_table_name = _get_new_table_name(DB)
    assert new_table_name not in DB.ls()

    new_table = DB[new_table_name]
    assert isinstance(new_table, D4M.db.DbTable)
    assert new_table.name == new_table_name
    assert new_table.name in DB.ls()

    D4M.db.delete_table(new_table, force=True)
    assert new_table.name not in DB.ls()


def test_index_and_delete_pair(DB):
    new_table_name, new_table_nameT = _get_new_tablepair_name(DB)
    assert new_table_name not in DB.ls()
    assert new_table_nameT not in DB.ls()

    new_table_pair = D4M.db.get_index(DB, new_table_name, new_table_nameT)
    assert isinstance(new_table_pair, D4M.db.DbTablePair)
    assert new_table_pair.name_1 == new_table_name and new_table_pair.name_1 in DB.ls()
    assert new_table_pair.name_2 == new_table_nameT and new_table_pair.name_2 in DB.ls()

    D4M.db.delete_table(new_table_pair, force=True)
    assert new_table_pair.name_1 not in DB.ls() and new_table_pair.name_2 not in DB.ls()


def test_index_and_delete_pair_getitem(DB):
    new_table_name, new_table_nameT = _get_new_tablepair_name(DB)
    assert new_table_name not in DB.ls()
    assert new_table_nameT not in DB.ls()

    new_table_pair = DB[new_table_name, new_table_nameT]
    assert isinstance(new_table_pair, D4M.db.DbTablePair)
    assert new_table_pair.name_1 == new_table_name and new_table_pair.name_1 in DB.ls()
    assert new_table_pair.name_2 == new_table_nameT and new_table_pair.name_2 in DB.ls()

    D4M.db.delete_table(new_table_pair, force=True)
    assert new_table_pair.name_1 not in DB.ls() and new_table_pair.name_2 not in DB.ls()


def test_put_triples(DB):
    new_table_name = _get_new_table_name(DB)
    new_table = D4M.db.get_index(DB, new_table_name)
    assert D4M.db.nnz(new_table) == 0
    D4M.db.put_triple(new_table, "1,2,3,4,", "A,B,C,D,", "A1,B2,C3,D4,")
    assert D4M.db.nnz(new_table) == 4
    D4M.db.delete_table(new_table, force=True)


@pytest.mark.parametrize(
    "row_query,col_query",
    [
        (slice(None, None), slice(None, None)),
        ("a,b,", "A,B,"),
        ("1,5,", "8,A,"),
        ("d,e,", "D,E,"),
        ("1,:,7,", "2,4,"),
        (["a", "b", "d"], ["A", "C", "D"]),
        (":", ":"),
    ],
)
def test_index_assoc_single(row_query, col_query, table, row, col, val):
    test_assoc = D4M.assoc.Assoc(row, col, val)
    test_assoc[row_query, col_query].printfull()
    D4M.db.get_index(table, row_query, col_query).printfull()

    assert D4M.assoc.assoc_equal(
        test_assoc[row_query, col_query],
        D4M.db.get_index(table, row_query, col_query),
        return_info=True,
    )


@pytest.mark.parametrize(
    "row_query,col_query",
    [
        (slice(None, None), slice(None, None)),
        ("a,b,", "A,B,"),
        ("1,5,", "8,A,"),
        ("d,e,", "D,E,"),
        ("1,:,7,", "2,4,"),
        (["a", "b", "d"], ["A", "C", "D"]),
        (":", ":"),
    ],
)
def test_index_assoc_single_getitem(row_query, col_query, table, row, col, val):
    test_assoc = D4M.assoc.Assoc(row, col, val)
    test_assoc[row_query, col_query].printfull()
    D4M.db.get_index(table, row_query, col_query).printfull()

    assert D4M.assoc.assoc_equal(
        test_assoc[row_query, col_query],
        table[row_query, col_query],
        return_info=True,
    )


@pytest.mark.parametrize(
    "row_query,col_query",
    [
        (slice(None, None), slice(None, None)),
        ("a,b,", "A,B,"),
        ("1,5,", "8,A,"),
        ("d,e,", "D,E,"),
        ("1,:,7,", "2,4,"),
        (["a", "b", "d"], ["A", "C", "D"]),
        (":", ":"),
        (":", "A,B,"),
        (":", "8,A,"),
        (":", "D,E,"),
        (":", "2,4,"),
        (":", ["A", "C", "D"]),
    ],
)
def test_index_assoc_pair(row_query, col_query, tablepair, row, col, val):
    test_assoc = D4M.assoc.Assoc(row, col, val)
    test_assoc[row_query, col_query].printfull()
    D4M.db.get_index(tablepair, row_query, col_query).printfull()

    assert D4M.assoc.assoc_equal(
        test_assoc[row_query, col_query],
        D4M.db.get_index(tablepair, row_query, col_query),
        return_info=True,
    )


@pytest.mark.parametrize(
    "row_query,col_query",
    [
        (slice(None, None), slice(None, None)),
        ("a,b,", "A,B,"),
        ("1,5,", "8,A,"),
        ("d,e,", "D,E,"),
        ("1,:,7,", "2,4,"),
        (["a", "b", "d"], ["A", "C", "D"]),
        (":", ":"),
        (":", "A,B,"),
        (":", "8,A,"),
        (":", "D,E,"),
        (":", "2,4,"),
        (":", ["A", "C", "D"]),
    ],
)
def test_index_assoc_pair_getitem(row_query, col_query, tablepair, row, col, val):
    test_assoc = D4M.assoc.Assoc(row, col, val)
    test_assoc[row_query, col_query].printfull()
    D4M.db.get_index(tablepair, row_query, col_query).printfull()

    assert D4M.assoc.assoc_equal(
        test_assoc[row_query, col_query],
        tablepair[row_query, col_query],
        return_info=True,
    )


@pytest.mark.parametrize("num_limit", [2, 0, 1, 5])
def test_get_index_iter(num_limit, table, row, col, val):
    test_assoc = D4M.assoc.Assoc(row, col, val)
    row, col, val = (
        D4M.util.sanitize(row),
        D4M.util.sanitize(col),
        D4M.util.sanitize(val),
    )

    new_iterator = D4M.db.get_iterator(table, num_limit)

    if num_limit == 0:
        assert D4M.assoc.assoc_equal(test_assoc, D4M.db.get_index(new_iterator))
    else:
        index = 0
        while index + num_limit <= len(val):
            test_assoc = D4M.assoc.Assoc(
                row[index: index + num_limit],
                col[index: index + num_limit],
                val[index: index + num_limit],
            )
            if index == 0:
                assert D4M.assoc.assoc_equal(test_assoc, D4M.db.get_index(new_iterator, ":", ":"))
            else:
                assert D4M.assoc.assoc_equal(test_assoc, D4M.db.get_index(new_iterator))
            index += num_limit
        if index < len(val):
            test_assoc = D4M.assoc.Assoc(row[index:], col[index:], val[index:])
            assert D4M.assoc.assoc_equal(test_assoc, D4M.db.get_index(new_iterator))

        # Run one last time to check that iterator returns empty array
        assert D4M.assoc.assoc_equal(D4M.assoc.Assoc([], [], []), D4M.db.get_index(new_iterator))
