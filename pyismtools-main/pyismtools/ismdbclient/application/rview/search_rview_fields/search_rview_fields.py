from pyismtools.utils2 import func as fu
from pyismtools.ismdbclient.application.protocols.rview import Rview

def search_rview_fields(rview: Rview, filters, limit = 5, 
                        offset = 0, 
                        cache: fu.FIFOKVCacheProtocol = None):

    return rview.fields_asearch(filters, limit, offset, cache)
