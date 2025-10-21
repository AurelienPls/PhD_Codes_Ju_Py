import pyismtools.utils2.domain.model2 as m
from pyismtools.ismdbclient.domain.model.view import view as dv

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

def subview(view, fields_ids):
    v = dv.subview(view, fields_ids)
    return create(schema(v), data(v))

