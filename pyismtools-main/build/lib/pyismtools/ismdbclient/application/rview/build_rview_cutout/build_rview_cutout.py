from pyismtools.ismdbclient.domain.model.view import view as dv
from pyismtools.ismdbclient.domain.model.view import schema as ds
from pyismtools.ismdbclient.application.view import schema
from pyismtools.ismdbclient.application.view import view
from pyismtools.ismdbclient.application.protocols.rview import Rview

def build_rview_cutout(rview: Rview, select, where):
    v = rview.sync_cutout_stream(select, where)
    vs = dv.schema(v)
    nvs = schema.create(ds.fields(vs), 
                        ds.entityfield(vs),
                        ds.utypes(vs))

    return view.create(nvs, dv.data(v))

