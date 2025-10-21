import re
import hashlib
import json
from pyismtools.utils2 import net as un

from pyismtools.ismdbclient.domain.model.view import view as v
from pyismtools.ismdbclient.domain.model.view import field as vfield
from pyismtools.ismdbclient.domain.model.view import schema as vschema

DEFAULT_ASEARCH_LIMIT = 5
DEFAULT_CUTOUT_PAGE_SIZE = 500
DEFAULT_CUTOUT_HEADERS = 2 # number of rows used as header in a cutout result.

class Rview:
    def __init__(self, uri: str) -> None:
        self.uri = uri

    def vparameters(self, filters, include_meta = True):
        return vparameters(self, filters, include_meta)

    def field_meta(self, field):
        return field_meta(self, field)

    def fields_asearch(self, filters, limit = DEFAULT_ASEARCH_LIMIT, 
                       offset = 0, cache = None):
        return fields_asearch(self, filters, limit, offset, cache)

    def sync_cutout(self, select, where, limit = DEFAULT_CUTOUT_PAGE_SIZE, 
                    offset = 0):
        return sync_cutout(self, select, where, limit, offset)

    def sync_cutout_stream(self, select, where,  
           page_size = DEFAULT_CUTOUT_PAGE_SIZE, 
           headers = DEFAULT_CUTOUT_HEADERS):
        return sync_cutout_stream(self, select, where, page_size, 
                                  headers)

    def entity(self, id, select = []):
        return entity(self, id, select)

    def dataset(self, id, select = []):
        return entity(self, id, select)
    
    def schema(self):
        return schema(self)


def create(cf):
    return Rview(cf['uri'])


# implementation
# --------------

# ### utils

def _handle_api_errors(res):
    if res.get('errors', []):
        raise StandardError('\n'.join(res['errors']))

# ### protocol

def vparameters(rview, filters, include_meta):
    uri = '/'.join([rview.uri, 'vparameters'])
    filters = ["".join(map(str, [p, o, v])) for p, o, v in filters]
    r = un.call_uri(uri, 
            {'filter': filters, 'meta': 1 if include_meta else 0}, 
            method = 'GET')
    _handle_api_errors(r)
    print(r)
    return r['data']

def field_meta(rview, field):
    uri = '/'.join([rview.uri, 'fields'])
    return un.call_uri(uri, {'ident': field}, method = 'GET')

def fields_asearch(rview, filters, limit, offset, cache):
    """
    Advanced field search.

    returns a stream of (<field ident>, assoc list of (meta_name, meta_value)). 
    Each field is described by a set of (meta_name, meta_value)
    
        ex: ('field/name', 'Initial density')

    This allows a list of *filter* to be defined to restrict the result set.
    
    A *filter* is a tuple (*att*, *op*, *val*).
    *att* is any valid attribute of the fields, as defined by the view schema.
    *op* is one of `=` `<` `>` `like` `ilike`
    *val* is any text/num value as relevant for the given (*attribute*, *op*)
    """

    uri = '/'.join([rview.uri, 'fields/asearch'])
    params = {
            'filters': filters,
            'limit': limit,
            'offset': offset
            }

    data = []
    errors = []

    if cache is not None:
        key = hashlib.sha1(json.dumps([view,
                                       filters,
                                       limit,
                                       offset]).encode()).hexdigest()
        if cache('contains', key):
            data = cache('get', key)
        else:
            r = _call_api(uri, params)
            _handle_api_errors(r)
            cache('add', key, r['data'])
    else:
        r = un.call_uri(uri, params, method = 'POSTJSON')
        _handle_api_errors(r)
        data = r['data']
    
    return data

def sync_cutout(rview, select, where, limit, offset):
    """
    Returns a *cutout* (subset) of the view `view` consisting of only the fields
    present in `select` for the elements (rows/objects) satisfying the
    constraints in `where`.

    The format of a *cutout* is a dict:
        
        {fields: <fields>, data: <data>}

    `fields` is (cselect, fields_meta)
        `fields_meta` is a list of (<field ident>, <field meta>)

    `data` is a list of

        (value of fields for element 1,
         ...                 element 2,
         ...)
    
    `where` is a list of *constraint*s
    
    a *constraint* is a dict {:att, :op, :val} holding respectively
    (attribute, operator, value), for example 
    
        {":att": "field/pubid", ":op": "=", ":val": "c/nh_init"}.

    The `attribute`s available are found in the view schema.
   
    `inc_entity` indicates if the ident of the elements (represented by a view
    row) must be returned as an additional cutout field.
   
   """
   
    uri = '/'.join([rview.uri, 'cutout'])
    params = {
            'select' : select, 
            'where' : where,
            'inc_entity': 1,
            'limit': limit, 
            'offset': offset
            }
    res = un.call_uri(uri, params, method = 'POSTJSON')
    _handle_api_errors(res)
    return res['data']

def sync_cutout_stream(rview, select, where, page_size, headers):

    """ returns the cutout as a stream of rows whose 2 first are 1. list of
    field, 2. list of dict with metadata about each field """
        
    def _stream_adapter(limit, offset):
        data = sync_cutout(rview, select, where, limit, offset) 

        fields, data = [data[k] for k in ('fields', 'data')]
        cselect, fields_meta = fields
        data = [cselect, fields_meta] + data
        return data

    _stream = un.paginated_stream_proxy(
            lambda l, o: _stream_adapter(l, o),
            page_size = page_size, 
            headers = headers)
   
    cutout = list(_stream)
    cutout_header = [dict(e) for eid, e in cutout[1]]

    fields = [vfield.create(ctf.get('field/name', ctf.get('ID')), 
                        id = ctf['ID'],
                        unit = ctf.get('field/unit', ''), 
                        utype = ctf.get('field/utype', ''), 
                        label = ctf.get('field/label', '')) 
              for ctf in cutout_header]

    view = v.create(vschema.create(fields, entityfield = 'entity'), 
                    cutout[2:])
    return view

def entity(rview, id, select):
    """ 
    returns all the selected fields of a view for a given element (identified
    by `ident`)

    If `select` is empty, all the fields for the element are returned.
    """
    uri = '/'.join([rview.uri, 'entity'])
    params = {'select' : select, 'ident': id}
    res = un.call_uri(uri, params, method = 'GET')
    _handle_api_errors(res)
    return res['data']

def schema(rview):

    def txt_to_dict(txt_schema):
        def _emptyline(l):
            return not l.strip()
        d = {}
        for l in txt_schema.split('\n'):
            if _emptyline(l):
                continue
            e, a, v = [e.strip() for e in l.split('\t')]
            d[e] = d.get(e, []) + [(a, v)]
        return d

    uri = '/'.join([rview.uri, 'schema'])
    res = un.call_uri(uri, {}, method = 'GET')
    return txt_to_dict(res)


