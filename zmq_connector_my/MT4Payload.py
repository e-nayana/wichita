def init_main_payload(method=None):
    if(method is None):
        return {
            'method': 'PING',
            'data': {}
        }
    return {
        'method': method,
        'data': {}
    }


def init_simple_message():
    return {'message': 'Hello world'}


def init_order_open():
    return {'action': 'OPEN',
            'type': 0,
            'symbol': 'EURUSD',
            'price': 0.0,
            'SL': 500,
             'TP': 500,
            'comment': 'DWX_Python_to_MT',
            'lots': 0.01,
            'magic': 123456,
            'ticket': 0}


def get_payload(method=None, data=None):
    payload = init_main_payload(method)
    if data is None:
        payload['data'] = init_simple_message()
        return payload
    payload["data"] = data
    return payload


