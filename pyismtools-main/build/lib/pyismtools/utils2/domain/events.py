def create(event_type, data):
    
    return {
        ':type': event_type,
        'data': data
        }

def etype(event):
    return event[':type']

def data(event):
    return event['data']

