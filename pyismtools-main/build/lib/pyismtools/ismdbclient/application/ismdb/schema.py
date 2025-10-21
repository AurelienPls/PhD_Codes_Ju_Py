import pyismtools.utils2.domain.model2 as m
from pyismtools.ismdbclient.domain.model.view import view as dv
from pyismtools.ismdbclient.domain.model.view import schema as dvs
from pyismtools.ismdbclient.domain.model.view import field as dvf


DEFAULT_UTYPES = {
        'parameter': 'SimDM:/resource/experiment/ParameterSetting.numericValue.value',
        'statistic': 'SimDM:/resource/experiment/StatisticalSummary.numericValue.value',
        'experiment': 'SimDM:/resource/experiment/Experiment.publisherDID',
        'dataset': 'SimDM:/resource/experiment/OutputDataset.publisherDID'
        }

class IsmdbViewSchema(m.Model):
    pass

def create(view: dv.View, utypes = None):

    def _add_dataset_field(schema):
        entity_field = dvs.entityfield(schema)
        for f in dvs.fields(schema):
            if m.id_(f) == entity_field:
                utype = DEFAULT_UTYPES['dataset']
                if utypes and 'dataset' in utypes:
                    utype = utypes['dataset']
                m.set_(f, 
                       {'id': 'datasetid', 'name': 'Dataset id', 'utype': utype})
                break

    def _add_model_field(schema):
        for f in dvs.fields(schema): 
            if dvf.utype(f) == experiment_utype(schema):
                m.set_(f, {'id': 'model', 'name': 'Model'})
                break

    vschema = dv.schema(view)
    _add_dataset_field(vschema) # change 'entity' -> 'dataset',

    _utypes = {k:v for k,v in DEFAULT_UTYPES.items()}
    if isinstance(utypes, dict):
        for k, val in utypes.items():
            _utypes[k] = val

    _ivs = IsmdbViewSchema({
            'fields': dvs.fields(vschema),
            'utypes': _utypes
            })

    _add_model_field(_ivs) # change 'experiment' -> 'model' 

    return _ivs

def parameter_utype(schema):
    return m.get(schema, 'utypes')['parameter']

def statistic_utype(schema):
    return m.get(schema, 'utypes')['statistic']

def experiment_utype(schema):
    return m.get(schema, 'utypes')['experiment']

def parameter_fields(schema):
    return dvs.utypes_filter(schema, utypes = [parameter_utype(schema)])

def statistic_fields(schema):
    return dvs.utypes_filter(schema, utypes = [statistic_utype(schema)])

def experiment_field(schema):
    return dvs.utypes_filter(schema, utypes = [experiment_utype(schema)])

def model_field(schema):
    return experiment_field(schema)

