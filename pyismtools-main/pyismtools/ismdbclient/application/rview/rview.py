from pyismtools.ismdbclient.application.protocols.rview import Rview

def fieldname_2_fieldid(rview: Rview, name):
    m = _fieldname_exists(rview, name)
    if m:
        fid, fmeta = m
        return fid
    return None

def _fieldname_exists(rview: Rview, name):
    try:
        r = rview.fields_asearch([['field/name', '=', name]])
        if r: 
            ident, meta = r[0]
            return (ident, dict(meta))
        return tuple()
    except Exception as e:
        print("exception in _exists_fieldname name: {}, e: {}".format(name, e))


