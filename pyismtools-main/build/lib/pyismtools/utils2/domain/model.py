from pyismtools.utils2 import utils as iu
from pyismtools.utils2.domain import events as ev


# default id function
# -------------------

def uid():
    import uuid
    return str(uuid.uuid4())


# exceptions
# ----------

class InternalProperty(Exception):
    pass

class ApiFunctionNotFound(Exception):
    pass


# internal
# --------

def _internal(property):
    return property[0] == ':'

def _iget(model, k):
    """ allow access to internal properties """
    return model[k]

def _iset(model, k, v):
    model[k] = v

# def _api_proxy(api, fname, model, data):
#     if fname not in api:
#         raise ApiFunctionNotFound
#     return api[fname](model, data)

class APIdict(dict):
    def __missing__(self, key):
        raise ApiFunctionNotFound


# events
# ------

def ev_property_changed(ovalue, nvalue):
    return ev.create('property_changed', 
                     {'oldvalue': ovalue, 'newvalue': nvalue})    



# api
# ---

def define(type_, inherits = [], idfunc = None):
    # WARNING !!!!
    # le fonctionnement de l'heritage ici ne fonctionne pas car il n'y a aucun
    # moyen de dire de quelle classe on parle, par exemple la classe mere peut
    # être "view" mais "view" peut être definie de plusieurs facons
    # differentes a plusieurs endroits du ccode....
    def _create(properties):
        model = {
                ':type': type_, 
                ':inherits': inherits, 
                ':events': [],
                ':api': APIdict()
                }
       
        if 'id' not in properties or not properties['id']:
            if idfunc:
                properties['id'] = idfunc()
            else:
                properties['id'] = uid()

        for k, v in properties.items():
            model[k] = v

        return model

    return _create

def extend_api(model, api):
    model_api = _iget(model, ':api')
    for k, v in api.items():
        model_api[k] = v
    _iset(model, ':api', model_api)
    
def api(model):
    return _iget(model, ':api')

def set_(model, properties, silent = False):
    for k, v in properties.items():
        if _internal(k):
            raise InternalProperty

        if k in model and not silent:
             oldvalue = get(model, k)
             register_event(model, ev_property_changed(oldvalue, v))
        model[k] = v

def properties(model):
    return {k:v for k, v in model.items() if not _internal(k)}

def clone(model):
    try:
        return api(model)['clone'](model)
    except ApiFunctionNotFound as e:
        return {k: v for k, v in model.items()}

def get(model, k, **kwargs):

    def _get(model, k, **kwargs):
        if _internal(k):
            raise InternalProperty

        if 'default' in kwargs:
            return model.get(k, kwargs['default'])
        
        return model[k]

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
    return _iget(model, ':type')

def isa_(model, type__):
    return type_(model) == type__ or inherits(model, type__)

def inherits(model, type_):
    return type_ in _iget(model, ':inherits')

def register_event(model, event):
    _events = registered_events(model)
    iu.dg('registering event {}'.format(event))
    _iset(model, ':events', _events + [event])

def registered_events(model):
    return [e for e in _iget(model, ':events')]

def clear_registered_events(model):
    _iget(model, ':events').clear()


