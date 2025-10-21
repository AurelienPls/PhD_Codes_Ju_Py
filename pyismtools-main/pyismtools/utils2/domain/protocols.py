import pyismtools.utils2.domain.model as m

class ProtocolNotDefinedForModel(Exception):
    pass

class ProtocolFunctionNotDefined(Exception):
    pass

class PROTOdict(dict):
    def __missing__(self, key):
        raise ApiFunctionNotFound

class PROTOFdict(dict):
    def __missing__(self, key):
        raise ProtocolFunctionNotDefined

def _get_pfunc(protocol, funcname):
    return protocol[funcname]

def _get_protocol(model, protocol):
    return m._iget(model, ':protocols')[protocol]

def _set_protocol(model, protocol, protocol_impl):
    try:
        protocols = m._iget(model, ':protocols')
    except:
        protocols = PROTOdict({})
    protocols[protocol] = protocol_impl
    m._iset(model, ':protocols', protocols)
    return model

def protocolfunc(model, protocol, funcname, *args):
    p = _get_protocol(model, protocol)
    return p[funcname](*args)

def extend(model, protocol, protocol_impl):
    _set_protocol(model, protocol, PROTOFdict(protocol_impl))
 
