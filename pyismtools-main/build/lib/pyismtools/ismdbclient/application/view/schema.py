import pyismtools.utils2.domain.model2 as m
from pyismtools.ismdbclient.domain.model.view import schema as ds

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
    return ds.fields(schema, pos, map_utypes)

