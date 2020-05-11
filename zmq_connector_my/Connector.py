import zmq
from time import sleep
from pandas import DataFrame, Timestamp
from threading import Thread
import zmq_connector_my.MT4Payload
from zmq_connector_my.MT4Payload import get_payload, init_order_open


class ConnectorCls:

    def __init__(self,
                 _client_id='DLabs_Python',
                 _host='localhost',
                 _protocol='tcp',
                 _port=32771,
                 _delimiter=';',
                 _verbose=False
                 ):

        self.active = True
        self.client_id = _client_id
        self.host = _host
        self.protocol = _protocol
        self.zqm_context = zmq.Context()
        self.url = self.protocol + "://" + self.host + ":"
        self.port = _port

        self.socket = self.zqm_context.socket(zmq.REQ)
        self.socket.connect(self.url + str(self.port))
        print("Socket connected")

    def ping(self):
        body = get_payload()
        print(body)
        self.socket.send_json(body)
        reply = self.socket.recv_json()
        print(reply)

    def new_order(self):
        init_order_open_data = init_order_open()
        body = get_payload('NEW_ORDER', init_order_open_data)
        print(body)

        self.socket.send_json(body)
        response = self.socket.recv_json()
        print(response)




