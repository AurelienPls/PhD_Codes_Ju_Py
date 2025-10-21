import pyismtools.classes

import numpy as np
import sys, traceback

from pyismtools.ismdbclient.application.ismdb.list_grid_varying_parameters import list_grid_varying_parameters
from pyismtools.ismdbclient.adapters.outside.rview.rest import rview as rvproto
from pyismtools.ismdbclient.application.ismdb.extract_subgrid import extract_subgrid
from pyismtools.ismdbclient.application.view import view as vi
from pyismtools.ismdbclient.application.ismdb import ismdb_view as iv
from pyismtools.ismdbclient.application.rview import rview as rv

class Suppressor():

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, exception_type, value, traceback):
        sys.stdout = self.stdout
        if exception_type is not None:
            # Do normal exception handling
            raise Exception(f"Got exception: {exception_type} {value} {traceback}")

    def write(self, x): pass

    def flush(self): pass


def get_grid_parameters(project_name, env = 'prod'):
   env_segment = '/' if env == 'prod' else '/dev/'
   grid_uri = "http://api.obspm.fr{}rviews/v1/projects/{}/views/cloudv".format(env_segment,
                                                                               project_name)
   rview = rvproto.create({'uri':grid_uri})
   
   with Suppressor():
       metadata = list_grid_varying_parameters.list_grid_varying_parameters(rview, [])['meta'][1:] # we exclude the first element, as it describe the "experiment" name. We only want the parameters.
       
   # keep only the human names and the list of values for each parameter
   parameter_name_list = [ [x[1:] for x in meta if x[0]=='field/name'][0][0] for meta in metadata]
   parameter_values_list = [[x[1:] for x in meta if x[0]=='field/values'][0] for meta in metadata]
   return parameter_name_list, parameter_values_list
   
def get_grid(project_name,observable_names, env = 'prod'):
   env_segment = '/' if env == 'prod' else '/dev/'
   grid_uri = "http://api.obspm.fr{}rviews/v1/projects/{}/views/cloudv".format(env_segment,
                                                                               project_name)
   rview = rvproto.create({'uri':grid_uri})
   
   with Suppressor():
       metadata = list_grid_varying_parameters.list_grid_varying_parameters(rview, [])['meta'][1:] # we exclude the first element, as it describe the "experiment" name. We only want the parameters.
       
   # keep only the human names and the list of values for each parameter
   param_names = [ [x[1:] for x in meta if x[0]=='field/name'][0][0] for meta in metadata]
   
   select = param_names + observable_names
   iselect = [rv.fieldname_2_fieldid(rview, n) for n in select]
   
   with Suppressor():
       sg = extract_subgrid.extract_subgrid(rview, iselect, [])
   
   parameter_tab = np.array(vi.data(iv.parameters_view(sg)),dtype='float')
   observable_tab = np.array(vi.data(iv.statistics_view(sg)),dtype='float')
   
   return pyismtools.classes.Model_grid(project_name,param_names,observable_names,parameter_tab,observable_tab)
