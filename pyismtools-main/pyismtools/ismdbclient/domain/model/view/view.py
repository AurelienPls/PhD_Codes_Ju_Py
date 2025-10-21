import pyismtools.utils2.domain.model2 as m
from . import schema as dschema

class View(m.Model):
    pass

def create(schema_, data_, id = None):
    _view = View({
            'id': id,
            'schema': schema_,
            'data': data_
            })

    return _view

def schema(view):
    return m.get(view, 'schema')

def data(view):
    return m.get(view, 'data')

def filter_data(view, fields_ids):
    """ returns list of list representation of a view with only the fields"""
    import numpy as np

    s = schema(view)
    d = np.array(data(view))
    idx = [dschema.index(s, 'id', fi) for fi in fields_ids]
    return d[:, idx].tolist()

def subview(view, fields_ids):
    s = schema(view)
    d = data(view)

    idx = [dschema.index(s, 'id', fi) for fi in fields_ids]
    sv_fields = dschema.fields(s, pos = idx)
    sv_schema = dschema.create(sv_fields, utypes = dschema.utypes(s))
    sv_data = filter_data(view, fields_ids)

    return create(sv_schema, sv_data, id = m.id_(view))


