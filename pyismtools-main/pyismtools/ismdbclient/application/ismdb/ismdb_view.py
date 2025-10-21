import pyismtools.utils2.domain.model2 as m
from pyismtools.ismdbclient.application.view import view as v
from pyismtools.ismdbclient.application.ismdb import schema as ivs

class IsmdbView(m.Model):
    pass

def create(view: v.View, id = None, utypes = None):
    _iview = IsmdbView({
            'id': id,
            'schema': ivs.create(view, utypes),
            'data': v.data(view)
            })

    return _iview

def schema(view: IsmdbView):
    return v.schema(view)

def parameters_view(view: IsmdbView):
    param_fields_ids = [m.id_(f) 
                        for f in ivs.parameter_fields(v.schema(view))]
    return v.subview(view, param_fields_ids)

def statistics_view(view: IsmdbView):
    stat_fields_ids = [m.id_(f) 
                        for f in ivs.statistic_fields(v.schema(view))]
    return v.subview(view, stat_fields_ids)

