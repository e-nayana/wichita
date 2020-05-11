# from mql_zmq_mt4_connector.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
#
# _zmq = DWX_ZeroMQ_Connector()
#
# test = _zmq._DWX_MTX_GET_ALL_OPEN_TRADES_()
# print(test)
#
#
# _my_trade = _zmq._generate_default_order_dict()
#
# print(_my_trade)
#
#
# _my_trade['_lots'] = 0.01
#
# _my_trade['_SL'] = 250
#
# _my_trade['_TP'] = 750
#
# _my_trade['_comment'] = 'test 123456765432123456'
#
# print(_zmq._DWX_MTX_NEW_TRADE_(_order=_my_trade))



from zmq_connector_my.Connector import ConnectorCls

test = ConnectorCls()

test.new_order()