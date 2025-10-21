import re
import hashlib
import json

from pyismtools.utils2 import net as un

from pyismtools.ismdbclient.domain.model.view import view as v
from pyismtools.ismdbclient.domain.model.view import field as vfield
from pyismtools.ismdbclient.domain.model.view import schema as vschema

DEFAULT_ASEARCH_LIMIT = 5
DEFAULT_CUTOUT_PAGE_SIZE = 100
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
                    offset = 0, inc_entity = True):
        return sync_cutout(self, select, where, limit, offset, inc_entity)

    def sync_cutout_stream(self, select, where, 
           inc_entity = True, 
           page_size = DEFAULT_CUTOUT_PAGE_SIZE, 
           headers = DEFAULT_CUTOUT_HEADERS):
        return sync_cutout_stream(self, select, where, inc_entity, page_size, 
                                  headers)

    def entity(self, id, select):
        return entity(self, id, select)
    
    def schema(self):
        return schema(self)

def create(cf):
    return Rview(cf['uri'])


# implementation
# --------------

# ### utils

def _handle_api_errors(res):
    if res['errors']:
        raise StandardError('\n'.join(res['errors']))

# ### protocol

def vparameters(rview, filters, include_meta):
    return []

def field_meta(rview, field):
    return []
        
def fields_asearch(rview, filters, limit, offset, cache):
    return []

def sync_cutout(rview, select, where, limit, offset, inc_entity):
    from collections import OrderedDict
    res ={ 'fields':
          [
            ['entity', 'experiment', 'avmax', 'value/inta_00_13c_o_j10__j9', 'value/cd_h2'],
            [['entity', OrderedDict([('ID', 'entity')])], ['experiment', OrderedDict([(':db/datatype', 'string'), (':db/doc', 'the id of the experiment'), ('field/name', 'experiment'), ('field/utype', 'SimDM:/resource/experiment/Experiment.publisherDID'), ('ID', 'experiment')])], ['avmax', OrderedDict([(':db/datatype', 'real'), (':db/doc', '<b> AVmax </b> <br/> Size of the cloud expressed as a visual extinction<br/> <i>Unit: magnitude</i>'), ('field/absolute_order', '4'), ('field/display', '1'), ('field/groupid', 'Grp_General'), ('field/name', 'AVmax'), ('field/order_in_group', '4'), ('field/rvalmax', '100.0'), ('field/rvalmin', '1E-4'), ('field/unit', 'mag'), ('field/utype', 'SimDM:/resource/experiment/ParameterSetting.numericValue.value'), ('field/values', '2.00000000E+000, 2.00000000E+001, 1.00000000E+001, 7.00000000E+000, 4.00000000E+001, 5.00000000E+000, 3.00000000E+001, 1.00000000E+000'), ('ID', 'avmax')])], ['value/inta_00_13c_o_j10__j9', OrderedDict([(':db/datatype', 'real'), (':db/doc', 'Line intensity integrated over line profile and for a specific angle between the front of the PDR and the line of sight'), ('field/label', 'http://purl.obspm.fr/vocab/PhysicalQuantities/Intensity'), ('field/name', 'I(13CO J=10->J=9 angle 00 deg)'), ('field/objecttype', 'cloud'), ('field/stat', 'value'), ('field/unit', 'erg cm-2 s-1 sr-1'), ('field/utype', 'SimDM:/resource/experiment/StatisticalSummary.numericValue.value'), ('field/valmax', '5.806606e-05'), ('field/valmin', '1.266257e-25'), ('ID', 'value/inta_00_13c_o_j10__j9')])], ['value/cd_h2', OrderedDict([(':db/datatype', 'real'), (':db/doc', 'Column density'), ('field/label', 'http://purl.obspm.fr/vocab/PhysicalQuantities/ColumnDensityOfUFHFLCQGNIYNRP-UHFFFAOYSA-N'), ('field/name', 'N(H2)'), ('field/objecttype', 'cloud'), ('field/stat', 'value'), ('field/unit', 'cm-2'), ('field/utype', 'SimDM:/resource/experiment/StatisticalSummary.numericValue.value'), ('field/valmax', '3.741921e+22'), ('field/valmin', '97908860000.0'), ('ID', 'value/cd_h2')])]]
    ],
    'data': [
    ['P154G3_P_r1e0A1e1P1e10_s_20_cloud', 'P154G3_P_r1e0A1e1P1e10', 10.0, 1.602144e-13, 9.354484e+21], 
    ['P154G3_P_r1e0A1e1P1e11_s_20_cloud', 'P154G3_P_r1e0A1e1P1e11', 10.0,
     1.645369e-13, 9.35446e+21], 
    ['P154G3_P_r1e0A1e1P1e3_s_20_cloud', 'P154G3_P_r1e0A1e1P1e3', 10.0,
     1.657307e-14, 8.679173e+21], 
    ['P154G3_P_r1e0A1e1P1e4_s_20_cloud', 'P154G3_P_r1e0A1e1P1e4', 10.0,
     6.980765e-15, 9.319466e+21], 
    ['P154G3_P_r1e0A1e1P1e5_s_20_cloud', 'P154G3_P_r1e0A1e1P1e5', 10.0, 1.350123e-13, 9.352782e+21], ['P154G3_P_r1e0A1e1P1e6_s_20_cloud', 'P154G3_P_r1e0A1e1P1e6', 10.0, 7.496956e-12, 9.354692e+21], ['P154G3_P_r1e0A1e1P1e7_s_20_cloud', 'P154G3_P_r1e0A1e1P1e7', 10.0, 1.639423e-11, 9.354731e+21], ['P154G3_P_r1e0A1e1P1e8_s_20_cloud', 'P154G3_P_r1e0A1e1P1e8', 10.0, 2.333474e-12, 9.354592e+21], ['P154G3_P_r1e0A1e1P1e9_s_20_cloud', 'P154G3_P_r1e0A1e1P1e9', 10.0, 2.804585e-12, 9.354523e+21], ['P154G3_P_r1e0A1e1P1p9e10_s_20_cloud', 'P154G3_P_r1e0A1e1P1p9e10', 10.0, 1.56594e-13, 9.354477e+21], ['P154G3_P_r1e0A1e1P1p9e7_s_20_cloud', 'P154G3_P_r1e0A1e1P1p9e7', 10.0, 4.829642e-12, 9.354694e+21]
]}

    return res

def sync_cutout_stream(rview, select, where, inc_entity, page_size, headers):
    
    cutout = sync_cutout(rview, select, where, inc_entity, page_size, headers)
    cfields, cdata = [cutout[k] for k in ('fields', 'data')]
    cutout_header = [dict(e) for eid, e in cfields[1]]

    fields = [vfield.create(ctf.get('field/name', 'ID'), 
                        id = ctf['ID'],
                        unit = ctf.get('field/unit', ''), 
                        utype = ctf.get('field/utype', ''), 
                        label = ctf.get('field/label', '')) 
              for ctf in cutout_header]

    view = v.create(vschema.create(fields, entityfield = 'entity'), 
                    cdata)
    return view

def entity(rview, id, select):
	return []

def schema(rview):
	return []
	

# utils
# -----

def _handle_api_errors(res):
    if res['errors']:
        raise StandardError('\n'.join(res['errors']))


