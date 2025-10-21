from pyismtools.utils2 import utils as iu
from pyismtools.utils2.domain import events as ev
from pyismtools.utils2.domain import protocols as dproto

# default id function
# -------------------

def uid():
    import uuid
    return str(uuid.uuid4())


# exceptions
# ----------

class InternalProperty(Exception):
    pass


# events
# ------

def ev_property_changed(ovalue, nvalue):
    return ev.create('property_changed', 
                     {'oldvalue': ovalue, 'newvalue': nvalue})    

# api
# ---

class ModelBase:
    def __init__(self, properties):
        if properties.get('id', None) == None:
            properties['id'] = ModelBase._idfunc()

        for k, v in properties.items():
            setattr(self, k, v)

        self._events = []
        self._protocols = dproto.PROTOdict()

ModelBase._idfunc = classmethod(lambda self: uid()) 

class Model(ModelBase):
    pass

def define(type_, inherits = (), idfunc = uid, definitions = ""):

    def get_calling_module():
        import inspect
        frm = inspect.stack()[2]
        return inspect.getmodule(frm[0]).__name__

    def init(self, attributes):
        if attributes.get('id', None) == None:
            attributes['id'] = idfunc()
        # super(T, self).__init__(attributes)
        super(type(self), self).__init__(attributes)

    definitions = dict(definitions)
    if '__init__' not in definitions:
        definitions['__init__'] = init

    definitions['__module__'] = get_calling_module()
    return type(type_, (ModelBase,) + inherits, definitions)


# internal
# --------

def _internal(property):
    return property[0] == '_'

def _iget(model, k):
    """ allow access to internal properties """
    return getattr(model, k)

def _iset(model, k, v):
    setattr(model, k, v)


# protocol
# --------


def extend_protocol(model, protocol, protocol_impl):
    mprotocols = _iget(model, '_protocols')
    mprotocols[protocol] = dproto.PROTOFdict(protocol_impl)
    _iset(model, '_protocols', mprotocols)
    
def protocols(model):
    return _iget(model, '_protocols')

def set_(model, properties, silent = False):
    for k, v in properties.items():
        if _internal(k):
            raise InternalProperty

        if hasattr(model, k) and not silent:
             oldvalue = get(model, k)
             register_event(model, ev_property_changed(oldvalue, v))
        
        setattr(model, k, v)

def get(model, k, **kwargs):

    def _get(model, k, **kwargs):
        if _internal(k):
            raise InternalProperty

        if 'default' in kwargs:
            return getattr(model, k, kwargs['default'])
        
        return getattr(model, k)

    def _get_multiples(model, k, **kwargs):
        r = []
        for ki in k:
            r.append(_get(model, ki, **kwargs))
        return r

    if isinstance(k, list):
        return _get_multiples(model, k, **kwargs)

    return _get(model, k, **kwargs)

def id_(model):
    return get(model, 'id')

def type_(model):
    return type(model)

def properties(model):
    return {k:v for k, v in model.__dict__.items() if not _internal(k)}

def isa_(model, type__):
    return isinstance(model, type__)

def inherits(model, type_):
    return isa_(model, type_)

def register_event(model, event):
    # add event timestamp ?
    _events = registered_events(model)
    iu.dg('registering event {}'.format(event))
    _iset(model, '_events', _events + [event])

def registered_events(model):
    return [e for e in _iget(model, '_events')]

def clear_registered_events(model):
    _iget(model, '_events').clear()

