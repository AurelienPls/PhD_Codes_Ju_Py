from pyismtools.ismdbclient.application.ismdb import ismdb_view
from pyismtools.ismdbclient.application.protocols.ismdb_rview import IsmdbRview
from pyismtools.ismdbclient.application.rview.build_rview_cutout import build_rview_cutout
from pyismtools.ismdbclient.application.rview.search_rview_fields import search_rview_fields

EXPERIMENT_UTYPE = 'SimDM:/resource/experiment/Experiment.publisherDID'

def extract_subgrid(rview: IsmdbRview, select, where, utypes = None):

    efield = search_rview_fields.search_rview_fields(rview, 
                                                     [['field/utype', '=', EXPERIMENT_UTYPE]], 
                                                     limit = 1)
    efid = None
    if efield:
        efid, _ = efield[0]
        select = [efid] + select

    v = build_rview_cutout.build_rview_cutout(rview, select, where)

    return ismdb_view.create(v, utypes = utypes)

