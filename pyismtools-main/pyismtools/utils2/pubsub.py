#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import random

# -- utils

# - pubsub - publish/subscribe 

def hub_builder():

    hub = {'listener-topic' : {}}

    dispatcher = {
        'publish' : lambda topic, data: publish(hub, topic, data),
        'subscribe' : lambda topic, listener: subscribe(hub, topic, listener),
        'unsubscribe' : lambda subscribe_id: unsubscribe(hub, subscribe_id)
    }

    def dispatch(service, *args):
        return dispatcher[service](*args)
    
    return dispatch


def _publish_data(hub, topic, data):
    for listener in hub[topic].values():
        listener(data)
    return topic


def publish(hub, topic, data):
    if topic in hub:
        _publish_data(hub, topic, data)
        
    return topic


def _generate_subscription_id():
    hash_ = random.getrandbits(128)
    return "%032x" % hash_


def subscribe(hub, topic, listener):
    if topic not in hub:
        hub[topic] = {}

    subscribe_id = _generate_subscription_id()
        
    hub[topic][subscribe_id] = listener
    # to have o(1) access for unsubscribe
    hub['listener-topic'][subscribe_id] = hub[topic]

    # print("hub : ", hub)
    
    return subscribe_id


def unsubscribe(hub, subscribe_id):
    del hub['listener-topic'][subscribe_id][subscribe_id]
    return True



def test():

    def listener1(data):
        print("listener 1 : ", data)

    def listener1a(data):
        print("listener 1a : ", data)

        
    def listener2(data):
        a,b = data['a'], data['b']
        print("listener2 {} + {} = {}".format(a, b, a + b))
        
    hub = hub_builder();

    p1 = hub('subscribe', 'queue1', listener1)
    p1a = hub('subscribe', 'queue1', listener1a)
    
    p2 = hub('subscribe', 'queue2', listener2)

    hub('publish', 'queue1', {'toto' : 'happy', 'picsou': ['riri', 'fifi']})
    hub('publish', 'queue2', {'a' : 7, 'b': 3})

    hub('unsubscribe', p1)
    hub('unsubscribe', p1a)
    hub('unsubscribe', p2)


# test()
    
