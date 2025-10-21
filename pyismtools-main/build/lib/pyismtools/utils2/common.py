import re
import json
import datetime as dt
import urllib
import collections
import os

import yaml


MAILSEP = ','
PLEVELS = ('DEBUG', 'INFO', 'ERROR')
PLEVEL = 'DEBUG'



# utils
# -----

def dtnow():
    return dt.datetime.now()

def tstamp(cfdb):
    return dtnow().strftime(cfdb['core']['tstamp_fmt'])

def dt2tstamp(cfdb, dtdate):
    return dtdate.strftime(cf['core']['tstamp_fmt'])

def tstamp2dt(cfdb, ts):
    return dt.datetime.strptime(ts, cfdb['core']['tstamp_fmt'])

def _plevel_enabled(level):
    return PLEVELS.index(level) >= PLEVELS.index(PLEVEL)

def pt(*args, level = 'DEBUG'):
    tpl = "{} {}: ".format(dt.datetime.now(), level)
    if _plevel_enabled(level):
        print(tpl, *args)

def pdebug(*args):
    pt(*args, level = 'DEBUG')

dg = pdebug

def perror(*args):
    pt(*args, level = 'ERROR')

def pinfo(*args):
    pt(*args, level = 'INFO')

def get_client_address(environ):
    try:
        return environ['HTTP_X_FORWARDED_FOR'].split(',')[-1].strip()
    except KeyError:
        return environ['REMOTE_ADDR']

def loadjson(path):
    cf = {} 
    with open(path) as cff:
        cf = json.load(cff)
    return cf

def loadyaml(path):
    cf_l = []
    with open(path) as cff:
        cf_l = list(yaml.safe_load_all(cff))
    return list(filter(None, cf_l))

def copy_dict(d):
    return {k:v for k, v in d.items()}

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def ga(al, key, default = None):
    for k, v in al:
        if key == k:
            return v
    return default

# def call_uri(uri, params = {}, ssluri = False):
#     
#     headers = {}
#     req = urllib.request.Request(uri, 
#             urllib.parse.urlencode(params).encode(), 
#             headers)
# 
#     # send the request
#     result = None # error
#     urlopen_ret = None
#     if ssluri:
#         import ssl
#         context = ssl.create_default_context()
#         f_res = urllib.request.urlopen(req, context = context)
#     else:
#          f_res = urllib.request.urlopen(req, context = context)
# 
#     json_data = f_res.read().decode('utf-8')
#     try: # is it json document ? -> test header no ?
#         result = json.loads(json_data,
#                 object_pairs_hook = collections.OrderedDict)
#     except Exception as e:
#         print("error while decofing json:", e)
#         result = json_data
# 
#     return result

def iterfile(fpath, mode = 'b'):
	with open(fpath, mode="r{}".format(mode)) as _f: 
		yield from _f 

def copy_obj(obj, deep = True):
    import copy
    if deep:
        return copy.deepcopy(obj)
    return copy.copy(obj)

def update_dict(dsrc, ddst, inplace = False, overwrite = True):
    if inplace:
        updated = ddst
    else:
        updated = copy_obj(ddst)

    for k, v in dsrc.items():
        if k in updated and not overwrite:
            continue
        updated[k] = v

    return updated

def fproxy(api):

    class ApiFunctionNotImplemented(Exception):
        pass

    def _fproxy(action, *args, **kwargs):
        if action in api:
            return api[action](*args, **kwargs)
        else:
            raise ApiFunctionNotImplemented()

    return _fproxy

