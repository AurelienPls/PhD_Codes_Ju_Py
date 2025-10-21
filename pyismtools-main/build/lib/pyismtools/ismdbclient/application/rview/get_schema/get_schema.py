from pyismtools.ismdbclient.application.protocols.rview import Rview

def get_schema(rview):
    return rview.schema()
