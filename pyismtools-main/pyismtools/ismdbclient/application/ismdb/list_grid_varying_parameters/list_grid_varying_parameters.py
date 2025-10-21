from pyismtools.ismdbclient.application.protocols.ismdb_rview import IsmdbRview

def list_grid_varying_parameters(rview: IsmdbRview, filters = [], 
                                 include_meta = True):
    r = rview.vparameters(filters, include_meta)
    return {'header': r[0], 'meta': r[1], 'data': r[2]}

