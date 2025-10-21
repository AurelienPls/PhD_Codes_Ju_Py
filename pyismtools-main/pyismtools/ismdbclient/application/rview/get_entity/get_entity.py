from pyismtools.ismdbclient.application.protocols.rview import Rview

def get_entity(rview: Rview, id, fields):
    return rview.entity(id, fields)

