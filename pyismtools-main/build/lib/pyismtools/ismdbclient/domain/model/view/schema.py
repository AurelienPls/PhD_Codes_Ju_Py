
from pyismtools.utils2 import utils as u
import pyismtools.utils2.domain.model2 as m
from . import field as df

class ViewSchema(m.Model):
    pass

def create(fields_, entityfield = 'entity', utypes = None):
    
    _vs = ViewSchema({
            'fields': fields_,
            'utypes': utypes,
            'entityfield': entityfield
            })

    return _vs

def fields(schema, pos = None, map_utypes = True):

    _fields = m.get(schema, 'fields')
    _utypes = m.get(schema, 'utypes')
   
    # if utypes mapping has been provided, we update the field utype to
    # reflect the mapping.
    if _utypes is not None and map_utypes:
        for f in _fields:
            if df.utype(f) in _utypes:
                df.utype(f, _utypes[df.utype(f)])

    if pos == None:
        return _fields
    else:
        return [_fields[i] for i in pos]

def entityfield(schema):
    return m.get(schema, 'entityfield')

def utypes(schema):
    return m.get(schema, 'utypes')

def index(schema, field_att, field_val):
    for i, f in enumerate(fields(schema)):
        if m.get(f, field_att) == field_val:
            return i
    raise ValueError("No such item in schema")

def indexes(schema, field_att, field_val):
    ret = []
    for i, f in enumerate(fields(schema)):
        if m.get(f, field_att) == field_val:
            ret += [i]

    return ret

def utypes_filter(schema, utypes = []):
    import functools
    try:
        idx = functools.reduce(lambda x, y: x + indexes(schema, 'utype', y),
                               utypes,
                               [])
        return fields(schema, pos = idx)
    except Exception as e:
        u.dg(e)
        return []

