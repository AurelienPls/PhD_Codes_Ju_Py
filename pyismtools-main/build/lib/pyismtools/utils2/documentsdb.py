import re

import yaml

RE_NS = re.compile(r'^[^/]+/(.+)$')

def loadyaml(path):
    cf_l = []
    with open(path) as cff:
        cf_l = list(yaml.safe_load_all(cff))
    return list(filter(None, cf_l))

def loadyaml_str(ystr):
    cf_l = list(yaml.safe_load_all(ystr))
    return list(filter(None, cf_l))

def _unns(d):
    return {RE_NS.sub(r'\1', k):v for k, v in d.items()}

def _ns(document):
    for k, v in document.items():
        return k.split('/')[0]

def filterns(documents, ns, unns = True):
    for d in documents:
        for k, v in d.items():
            if ns == _ns(k):
                if unns:
                    yield _unns(d)
                else:
                    yield d
                break

# get document in list iof documents
def gd(al, key = None, default = None, _all = False, 
        idfn = lambda x: x.get('ident', None)):
    def _copydict(e):
        return {k:v for k, v in e.items()}

    e_l = []
    for e in al:
        if key == None: # return the first document in the list
            return _copydict(e)

        if key == idfn(e):
            if not _all:
                return _copydict(e)
            e_l.append(_copydict(e))

    return e_l or default

def did(document, idfn = lambda x: x.get('ident', None)):
    return idfn(document)



# def load_confdb(cf_filepath, db = None, merge = False, reset = False):
#     _db = {}
#     if db != None:
#         _db = db
# 
#     documents = loadyaml(cf_filepath)
#     for d in documents:
#         ns = _ns(d)
# 
#         if reset:
#             _db[ns] = []
#             reset = False
# 
#         if ns not in _db:
#             _db[ns] = []
# 
#         if len(_db[ns]) == 1 and merge:
#             _db[ns][0].update(_unns(d))
#         else:
#             _db[ns].append(_unns(d))
# 
#     return _db


def load_documentsdb(ddb_str, db = None, merge = False, reset = False):
    _db = {}
    if db != None:
        _db = db

    documents = loadyaml_str(ddb_str)
    for d in documents:
        ns = _ns(d)

        if reset:
            _db[ns] = []
            reset = False

        if ns not in _db:
            _db[ns] = []

        if len(_db[ns]) == 1 and merge:
            _db[ns][0].update(_unns(d))
        else:
            _db[ns].append(_unns(d))

    return _db


# def load_documentsdb(ddb_filepath, db = None, merge = False, reset = False):
#     _db = {}
#     if db != None:
#         _db = db
# 
#     documents = loadyaml(ddb_filepath)
#     for d in documents:
#         ns = _ns(d)
# 
#         if reset:
#             _db[ns] = []
#             reset = False
# 
#         if ns not in _db:
#             _db[ns] = []
# 
#         if len(_db[ns]) == 1 and merge:
#             _db[ns][0].update(_unns(d))
#         else:
#             _db[ns].append(_unns(d))
# 
#     return _db

