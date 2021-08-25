# TODO: Figure out how to test locally
# import pytest
# from py4j.java_gateway import JavaGateway
#
# import D4M.assoc
# import D4M.util
# from D4M.util import replace_default_args
# import D4M.db
#
# @pytest.mark.parametrize('py4j_path,accumulo_path,graphulo_path,die_on_exit',
#                          [(None, None, None, None)
#                           ])
# def test_start_java_gateway(py4j_path, accumulo_path, graphulo_path, die_on_exit):
#     py4j_path, accumulo_path, graphulo_path, die_on_exit = replace_default_args(D4M.db.JavaConnector.start_java(),
#                                                                                 py4j_path=py4j_path,
#                                                                                 accumulo_path=accumulo_path,
#                                                                                 graphulo_path=graphulo_path,
#                                                                                 die_on_exit=die_on_exit)
#     assert isinstance(D4M.db.JavaConnector.start_java(py4j_path=py4j_path, accumulo_path=accumulo_path,
#                                                       graphulo_path=graphulo_path, die_on_exit=die_on_exit),
#                       JavaGateway)
#
#
# gateway = D4M.db.JavaConnector.start_java()
#
#
# @pytest.mark.parametrize('java_func,java_inputs,java_outputs',
#                          [(gateway.jvm.java.lang.math.pow, (2, 3), 8.0)
#                           ])
# def test_start_java_jvm(java_func, java_inputs, java_outputs):
#     assert java_func(*java_inputs) == java_outputs
