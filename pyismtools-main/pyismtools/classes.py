import numpy as np
import pyismtools.utils
import scipy.interpolate
import copy
from tqdm.auto import tqdm
import glob
from operator import methodcaller
import sys

##########################
# Model class
##########################

class Model:
   """
   Class used to store the results of a single model.

   Attributes
   ----------
   N_parameters : int
       Number of parameters in the model
   N_observables : int
       Number of predicted observable quantities in the model
   grid_id : str
       Id of the model grid from which the model was extracted ("Not in a grid" is used when the model was not taken from a model grid)
   code_version : str
       Version of the code used to produce the model
   parameter_names : list[str]
       List of the names of the parameters in the model
   parameter_values : list[float]
       List of the values of the parameters in the model (in the same order as in parameter_names)
   observable_names : list[str]
       List of the names of the predicted observable quantities in the model
   observable_values : list[float]
       List of the (unscaled) values of the predicted observable quantities in the model (in the same order as in parameter_names)
   scaling_factor : float
       Value of the scaling factor (representing geometrical effect) used in the model
       Note : the values contained in observable_values are the UNSCALED values. All methods outputing observable quantities apply the scaling before outputting.
   interpolated_model : bool
       False : the model is an actual model produced by the code, True : the model is an interpolation from a grid of models produced by the code.
   id : str or None
       Id of the model

   Class methods
   -------------
   from_output_folder(cls,folder_path,restricted_parameter_name_list=None,restricted_observable_name_list=None)
       Create a new model from the output folder of a model

   Methods
   -------
   summary()
       Print a summary of the object (identical to print(object))
       
   copy()
       Returns a deep copy of the object
       
   get_grid_id()
       Returns the ID of the grid from which the model was extracted
       
   get_code_version()
       Returns the version of the code used to produce the model
       
   get_N_parameters()
       Returns the number of parameters in the model
       
   get_N_observables()
       Returns the number of observables in the model
       
   get_parameter_names()
       Returns the list of the names of the parameters of the model
       
   get_parameter_vector()
       Returns the parameter values of the model
       
   get_scaling_factor()
       Returns the value of the scaling factor of the model (returns 1. if the model has no scaling factor)
   
   set_scaling_factor(scaling_factor)
       Sets the value of the scaling factor of the model
       
   get_observable_names()
       Returns the list of the names of the observables of the model
       
   get_observable_vector(scaled=True)
       Returns the observable values of the model (scaled by the scaling factor if scaled==True)
       
   get_parameter_value(parameter_name)
       Returns the value of a single parameter (specified by "parameter_name") of the model
       
   get_get_observable_value(observable_name,scaled=True)
       Returns the value of a single observable (specified by "observable_name") of the model (scaled by the scaling factor if scaled==True)
       
   reduce_parameters(new_parameter_name_list)
       Only keep as parameters of the model the parameters contained in "new_parameter_name_list" (this list should only contain names that are present in the model).
       
   reduce_observables(new_observable_name_list)
       Only keep as observables of the model the observables contained in "new_observable_name_list" (this list should only contain names that are present in the model).
   
   translate_names(translation)
       Change the names of the observables in the model according to the Translation object passed in argument
       
   is_interpolated()
       Returns True if the model has been created by interpolation from a grid of models, False if it is a real model resulting from a run of the code.
       
   is_scaled()
       Returns True is the model has a scaling factor (that is != 1) and False otherwise.
   """
   
   N_parameters : int               # Number of parameters in the model
   N_observables : int              # Number of predicted observable quantities in the model
   grid_id : str                    # Id of the model grid from which the model was extracted ("Not in a grid" is used when the model was not taken from a model grid)
   code_version : str               # Version of the code used to produce the model
   parameter_names : list[str]      # List of the names of the parameters in the model
   parameter_values : list[float]   # List of the values of the parameters in the model (in the same order as in parameter_names)
   observable_names : list[str]     # List of the names of the predicted observable quantities in the model
   observable_values : list[float]  # List of the (unscaled) values of the predicted observable quantities in the model (in the same order as in parameter_names)
   scaling_factor : float           # Value of the scaling factor (representing geometrical effect) used in the model
                                    # Note : the values contained in observable_values are the UNSCALED values. All methods outputing observable quantities apply the scaling before outputting.
   interpolated_model : bool        # False : the model is an actual model produced by the code, True : the model is an interpolation from a grid of models produced by the code.
   id : str | None = None           # Id of the model
   
   def __init__(self,parameter_name_list,parameter_value_list,observable_name_list,observable_value_list,code_version,grid_id,scaling_factor=None,interpolated_model=False, id = None):
      '''add doc'''
      assert len(parameter_name_list)==len(parameter_value_list), "Trying to create model with number of parameter values incompatible with number of parameter names"
      assert len(observable_name_list)==len(observable_value_list), "Trying to create model with number of observable values incompatible with number of observable names"
      assert (len(set(parameter_name_list)) == len(parameter_name_list)), "Trying to create model with repeated parameter name"
      assert (len(set(observable_name_list)) == len(observable_name_list)), "Trying to create model with repeated parameter name"
      
      self.parameter_names = parameter_name_list
      self.parameter_values = parameter_value_list
      self.observable_names = observable_name_list
      self.observable_values = observable_value_list
      self.N_parameters = len(parameter_name_list)
      self.N_observables = len(observable_name_list)
      self.grid_id = grid_id
      self.code_version = code_version
      self.scaling_factor = None
      self.interpolated_model = interpolated_model
      self.id = id

   @classmethod
   def from_output_folder(cls,folder_path,restricted_parameter_name_list=None,restricted_observable_name_list=None):
   
      ### 1 - Read the .stat file for observable quantities (integrated intensities and species column densities only).
      stat_file_path = glob.glob(folder_path+"/*20.stat")[0] # assumes there is only one file ending with "20.dat" in the folder
      stat_file = open(stat_file_path,'r')
      # first line gives the date
      line = stat_file.readline()
      # second line gives the code version
      code_version = stat_file.readline()
      # third line gives model name, fourth is empty
      line = stat_file.readline()
      line = stat_file.readline()
      observable_name_list = []
      observable_value_list = []
      while line!="":
         line = stat_file.readline()
         if line.startswith("value") :
            line = line.split('#')
            if line[1].strip().startswith("inta") or (line[1].strip().startswith("cd_") and not line[1].strip().startswith("cd_lev_")):
               observable_name_list.append(line[1].strip())
               observable_value_list.append(float(line[2]))
      
      ### 2 - Read the beginning of the .def file
      def_file_path = glob.glob(folder_path+"/*.def")[0] # assumes there is only one file ending with ".def" in the folder
      def_file = open(def_file_path,'r')
      for i in range(10):
         line = def_file.readline()
      line = def_file.readline()
      parameter_name_list = []
      parameter_value_list = []
      while not line.startswith("-----") :
         line = line.split("!")
         try:   # if the parameter has a numerical value, include it in the parameter list
            value = float(line[0])
            name = line[1].split(":")[0]
            parameter_name_list.append(name.strip())
            parameter_value_list.append(value)
         except ValueError: # otherwise (float(line[0]) returns a ValueError) do nothing (we cannot handle the non numerical parameters so we ignore them)
            pass
         line = def_file.readline()
      
      ### 3 - Prepare values for the other attributes
      grid_id = "Not in a grid"
      
      ### 4 - Create the model object and restrict to desired list of parameters and observables
      model = cls(parameter_name_list,parameter_value_list,observable_name_list,observable_value_list,code_version,grid_id)
      if restricted_parameter_name_list is not None :
         model.reduce_parameters(restricted_parameter_name_list)
      if restricted_observable_name_list is not None :
         model.reduce_observables(restricted_observable_name_list)
      
      return model
      
   def __repr__(self):
      lines = []
      opt_text = "(scaled) " if self.scaling_factor is not None else ""
      lines.append("---------------------------------------------------------")
      if not self.interpolated_model :
          lines.append("Model                                                    ")
      else :
          lines.append("Interpolated model                                       ")
      lines.append("                                                         ")
      lines.append(f"From grid : {self.grid_id:21}  code version : {self.code_version:12}")
      lines.append(f"Number of parameters : {self.N_parameters:<10d}  number of observables : {self.N_observables:<10d}")
      if self.scaling_factor is None :
          lines.append("No scaling factor.                                        ")
      else :
          lines.append("With scaling factor.")
      lines.append("                                                         ")
      lines.append("Parameters :                                             ")
      for i in range(self.N_parameters):
         lines.append(f"{self.parameter_names[i]:30} {self.parameter_values[i]:10.2E}")
      if self.scaling_factor is not None :
         lines.append(f"(scaling_factor                {self.scaling_factor:10.2E})")
      lines.append("                                                         ")
      scaling_factor = self.scaling_factor if self.scaling_factor is not None else 1.
      lines.append(f"Observables {opt_text:s}:                                            ")
      if (self.N_observables<12):
         for i in range(self.N_observables):
            lines.append(f"{self.observable_names[i]:30} {self.observable_values[i]*scaling_factor:10.2E}")
      else :
         for i in range(5):
            lines.append(f"{self.observable_names[i]:30} {self.observable_values[i]*scaling_factor:10.2E}")
         lines.append("...")
         for i in range(self.N_observables-5,self.N_observables):
            lines.append(f"{self.observable_names[i]:30} {self.observable_values[i]*scaling_factor:10.2E}")
      lines.append("---------------------------------------------------------")
      return "\n".join(lines)
   
   def summary(self):
      print(self)
      
   def copy(self):
      return copy.deepcopy(self)
      
   def get_grid_id(self):
      return self.grid_id
      
   def get_code_version(self):
      return self.code_version
      
   def get_N_parameters(self):
      return self.N_parameters
      
   def get_N_observables(self):
      return self.N_observables
      
   def get_parameter_names(self):
      return self.parameter_names.copy()
      
   def get_parameter_vector(self):
      return np.array(self.parameter_values).copy()
      
   def get_scaling_factor(self):
      return self.scaling_factor if self.scaling_factor is not None else 1.
      
   def set_scaling_factor(self,scaling_factor):
      self.scaling_factor = float(scaling_factor)
      
   def get_observable_names(self):
      return self.observable_names.copy()
      
   def get_observable_vector(self,scaled=True):
      vector = np.array(self.observable_values)
      if scaled==True and self.scaling_factor is not None :
          vector *= self.scaling_factor
      return vector.copy()
      
   def get_parameter_value(self,parameter_name):
      assert parameter_name in self.parameter_names, "Trying to access parameter value in model for parameter name not in model."
      index_param = self.parameter_names.index(parameter_name)
      return self.parameter_values[index_param]
      
   def get_observable_value(self,observable_name,scaled=True):
      assert observable_name in self.observable_names, "Trying to access observable value in model for observable name not in model."
      index_param = self.observable_names.index(observable_name)
      value = self.observable_values[index_param]
      if scaled==True and self.scaling_factor is not None :
         value *= self.scaling_factor
      return value
      
   def reduce_parameters(self,new_parameter_name_list):
      for parameter_name in new_parameter_name_list:
          assert parameter_name in self.parameter_names, "Trying to redefine the list of parameters in model, but \"%s\" is not in the initial parameter list of the model."%(parameter_name)
      index_list = []
      for parameter_name in new_parameter_name_list:
          index_list.append(self.parameter_names.index(parameter_name))
      self.parameter_names = [self.parameter_names[i] for i in index_list]
      self.parameter_values = [self.parameter_values[i] for i in index_list]
      self.N_parameters = len(self.parameter_names)
      
   def reduce_observables(self,new_observable_name_list):
      for observable_name in new_observable_name_list:
          assert observable_name in self.observable_names, "Trying to redefine the list of observables in model, but \"%s\" is not in the initial observable list of the model."%(observable_name)
      index_list = []
      for observable_name in new_observable_name_list:
          index_list.append(self.observable_names.index(observable_name))
      self.observable_names = [self.observable_names[i] for i in index_list]
      self.observable_values = [self.observable_values[i] for i in index_list]
      self.N_observables = len(self.observable_names)
      
   def translate_names(self,translation):
      test_forward = np.all(np.array([name in translation.forward_dict.keys() for name in self.observable_names]))
      test_reverse = np.all(np.array([name in translation.reverse_dict.keys() for name in self.observable_names]))
      assert test_forward or test_reverse, "Attempting to translate observable names but some names are not present in the translation dictionnary."
      # need to modify only : observable_names : list[str]
      new_observable_names = []
      if test_forward :
         for name in self.observable_names :
            new_observable_names.append(translation.forward_dict[name])
      else :
         for name in self.observable_names :
            new_observable_names.append(translation.reverse_dict[name])
      self.observable_names = new_observable_names
      
   def is_interpolated(self):
      return self.interpolated_model
      
   def is_scaled(self):
      return (self.scaling_factor is not None) and (self.scaling_factor!=1.)
      
##########################
# Model_grid class
##########################

class Model_grid:
   '''add doc'''
   
   # define attributes here
   grid_id : str
   code_version : str
   N_parameters : int
   N_observables : int
   N_models : int
   parameter_names : list[str]
   observable_names : list[str]
   parameter_tab : np.ndarray
   observable_tab : np.ndarray
   infos : str
   models_ids: list[str] | None = None
   
   def __init__(self,grid_id,parameter_names,observable_names,parameter_tab,observable_tab,code_version="Unknown code version",infos="No infos.", models_ids = None):
      assert parameter_tab.shape[0] == observable_tab.shape[0], "Parameter tab and observable tab don't contain the same number of models."
      self.grid_id = grid_id
      self.code_version = code_version
      self.N_parameters = len(parameter_names)
      self.N_observables = len(observable_names)
      self.N_models = parameter_tab.shape[0]
      self.parameter_names = parameter_names
      self.observable_names = observable_names
      self.parameter_tab = parameter_tab
      self.observable_tab = observable_tab
      self.infos = infos
      self.models_ids = models_ids
      # sanitation : if a parameter takes a single value in the grid, remove it from the list of varied parameters
      self.eliminate_fixed_parameters()
      
   def summary(self):
      print(self)
   
   @classmethod
   def from_ASCII_file(cls,grid_file_name,N_parameters,grid_id):
      # Read file header
      print("Starting to read header...")
      names = []
      code_version = ""
      infos = ""
      file = open(grid_file_name,'r')
      # move to the start of first header block
      line = file.readline()
      while len(line)!=0 and not line.startswith("##="):
         line = file.readline()
      # read first header block
      line = file.readline()
      while len(line)!=0 and not line.startswith("##="):
         infos += line
         if line.startswith("## Code"):
             code_version = line.split(':')[-1]
         line = file.readline()
      # read second header block
      line = file.readline()
      while len(line)!=0 and not line.startswith("#="):
         names.append(line.split("|")[-1].strip())
         line = file.readline()
      # end of header
      file.close()
      parameter_names = names[:N_parameters]
      observable_names = names[N_parameters:]
      # Load data tab
      print("Reading data array...")
      tab = np.loadtxt(grid_file_name)
      N_models = tab.shape[0]
      N_observables = tab.shape[1] - N_parameters
      parameter_tab = tab[:,:N_parameters]
      observable_tab = tab[:,N_parameters:]
      print("Model grid loaded successfully.")
      
      return cls(grid_id,parameter_names,observable_names,parameter_tab,observable_tab,code_version,infos)
      
   @classmethod
   def from_model_list(cls,model_list,grid_id,infos="No infos."):
      # create parameter name list, observable name list and code version
      parameter_names = model_list[0].get_parameter_names()
      observable_names = model_list[0].get_observable_names()
      code_version = model_list[0].get_code_version()
      # Sanity checks
      assert all(model.get_parameter_names()==parameter_names for model in model_list), "When creating a grid from a model list, all models must have the same list of parameter names."
      assert all(model.get_observable_names()==observable_names for model in model_list), "When creating a grid from a model list, all models must have the same list of observable names."
      assert all(model.get_code_version()==code_version for model in model_list), "When creating a grid from a model list, all models must have the same code version."
      # Prepare the rest of the attributes
      N_observables = len(observable_names)
      N_parameters = len(parameter_names)
      N_models = len(model_list)
      parameter_tab = np.stack([model.get_parameter_vector() for model in model_list])
      observable_tab = np.stack([model.get_observable_vector() for model in model_list])

      return cls(grid_id,parameter_names,observable_names,parameter_tab,observable_tab,code_version,infos)
      
   def save_to_ASCII_file(self,file_name):
      # Build header
      header =  "##==================================================================================\n"
      header += "## DataFile       : "+self.grid_id+"\n"
      header += "## Code           : "+self.code_version
      header += self.infos
      header += "##==================================================================================\n"
      for i,name in enumerate(self.parameter_names) :
         header += "# %i | "%i+name+"\n"
      for i,name in enumerate(self.observable_names) :
         header += "# %i | "%i+name+"\n"
      header += "#=================================================================================="
      # build full array and save to file
      full_tab = np.hstack([self.parameter_tab,self.observable_tab])
      ## debug
      print(self.parameter_tab.shape,self.observable_tab.shape,full_tab.shape)
      ##
      np.savetxt(file_name,full_tab,header=header,comments="")
   
   def __repr__(self):
      lines = []
      lines.append("---------------------------------------------------------")
      lines.append("Model grid                                               ")
      lines.append(f"Grid ID : {self.grid_id:21}  code version : {self.code_version:12}")
      lines.append(f"Number of models : {self.N_models:<6d} number of parameters : {self.N_parameters:<6d} number of observables : {self.N_observables:<6d}")
      lines.append("                                                         ")
      lines.append("Parameters :                                             ")
      lines.append("name                             min_value  max_value nb_values")
      for i in range(self.N_parameters):
         lines.append(f"{self.parameter_names[i]:30} {self.parameter_tab[:,i].min():10.2E} {self.parameter_tab[:,i].max():11.2E} {len(np.unique(self.parameter_tab[:,i])):<4d}")
      lines.append("                                                         ")
      lines.append("Observables :                                            ")
      lines.append("name                             min_value  max_value")
      if (self.N_observables<12):
         for i in range(self.N_observables):
            lines.append(f"{self.observable_names[i]:30} {self.observable_tab[:,i].min():10.2E} {self.observable_tab[:,i].max():10.2E}")
      else :
         for i in range(5):
            lines.append(f"{self.observable_names[i]:30} {self.observable_tab[:,i].min():10.2E} {self.observable_tab[:,i].max():10.2E}")
         lines.append("...")
         for i in range(self.N_observables-5,self.N_observables):
            lines.append(f"{self.observable_names[i]:30} {self.observable_tab[:,i].min():10.2E} {self.observable_tab[:,i].max():10.2E}")
      lines.append("                                                         ")
      lines.append("Additional infos :                                       ")
      lines.append("                                                         ")
      lines.append(self.infos)
      lines.append("---------------------------------------------------------")
      return "\n".join(lines)
      
   def summary(self):
      print(self)
      
   def copy(self):
      return copy.deepcopy(self)
      
   def get_N_parameters(self):
      return self.N_parameters
      
   def get_N_observables(self):
      return self.N_observables
      
   def get_N_models(self):
      return self.N_models
   
   def get_infos(self):
      return self.infos
      
   def get_code_version(self):
      return self.code_version
   
   def get_grid_id(self):
      return self.grid_id
   
   def get_parameter_names(self):
      return self.parameter_names.copy()

   def get_observable_names(self):
      return self.observable_names.copy()
      
   def get_parameter_array(self,parameter_list=None):
      temp_array = self.parameter_tab.copy()
      if parameter_list is not None :
         index_list = []
         for parameter_name in parameter_list:
            index_list.append(self.parameter_names.index(parameter_name))
         temp_array = temp_array[:,index_list]
      return temp_array
      
   def get_observable_array(self,observable_list=None):
      temp_array = self.observable_tab.copy()
      if observable_list is not None :
         index_list = []
         for observable_name in observable_list:
            index_list.append(self.observable_names.index(observable_name))
         temp_array = temp_array[:,index_list]
      return temp_array
      
   def get_unique_parameter_values(self):
      unique_value_dict = {}
      for i in range(self.N_parameters):
         unique_value_dict[self.parameter_names[i]] = list(np.unique(self.parameter_tab[:,i]))
      return unique_value_dict
   
   def get_model(self,parameter_vector):
      assert(len(parameter_vector) == self.N_parameters), "Trying to retrieve model from grid with a number of parameters different from the number of parameters in the grid."
      assert list(parameter_vector) in self.parameter_tab.tolist(), "Trying to retrieve model from grid with parameter vector that does not exist in grid."
      parameter_vector = np.array(parameter_vector) # converting in case user gave a list
      index_model = self.parameter_tab.tolist().index(list(parameter_vector))
      return Model(self.parameter_names.copy(),
                   list(parameter_vector),
                   self.observable_names.copy(),
                   list(self.observable_tab[index_model,:]),
                   self.code_version,
                   self.grid_id,
                   id = self.models_ids[index_model] if self.models_ids else None)
   
   def get_model_observable_vector(self,parameter_vector):
      parameter_vector = np.array(parameter_vector)
      model = self.get_model(parameter_vector)
      return model.get_observable_vector()

   def get_nearest_model(self,parameter_vector,distance_measure="log_euclidian"):
      assert(len(parameter_vector) == self.N_parameters), "Trying to retrieve model from grid with a number of parameters different from the number of parameters in the grid."
      assert distance_measure in ["euclidian","log_euclidian"], "Model_grid.get_nearest_model : Unknown distance measure %s"%(distance_measure)
      parameter_vector = np.array(parameter_vector) # converting in case user gave a list
      if distance_measure=="euclidian":
         distance = pyismtools.utils.distance_euclidian
      elif distance_measure=="log_euclidian" :
         distance = pyismtools.utils.distance_log_euclidian
      index_closest = np.argmin(distance(parameter_vector[np.newaxis,:],self.parameter_tab))
      return Model(self.parameter_names.copy(),list(self.parameter_tab[index_closest,:]),self.observable_names.copy(),list(self.observable_tab[index_closest,:]),self.code_version,self.grid_id)
   
   def eliminate_fixed_parameters(self):
      # sanitation : if a parameter takes a single value in the grid, remove it from the list of varied parameters
      param_value_dict = {} # Will contain the fixed parameters and their values
      for i,parameter in enumerate(self.parameter_names):
         if ( len(np.unique(self.parameter_tab[:,i])) == 1 ): # If one of the parameters takes only one value in the grid, remove it from the list of varied parameters
            param_value_dict[parameter] = np.unique(self.parameter_tab[:,i])[0]
      if len(param_value_dict)!=0 : # if some parameters need to be fixed, otherwise do nothing
         self.reduce_parameters(param_value_dict)
   
   def reduce_parameters(self,fixed_param_name_value_dict):
      for parameter in fixed_param_name_value_dict.keys():
         assert parameter in self.parameter_names, "Fixed parameter %s in not in initial parameter list of the grid."%parameter
         assert fixed_param_name_value_dict[parameter] in self.get_parameter_array()[:,self.parameter_names.index(parameter)], "Fixed parameter value for %s in not present the grid."%parameter

      # build index lists for fixed and free parameters
      fixed_index_list = []
      free_index_list = []
      for parameter_name in fixed_param_name_value_dict.keys() :
            fixed_index_list.append(self.parameter_names.index(parameter_name))
      for parameter_name in self.parameter_names:
         if parameter_name not in fixed_param_name_value_dict.keys() :
            free_index_list.append(self.parameter_names.index(parameter_name))
            
      # select models with right values of fixed parameters
      model_index_list = np.where((self.parameter_tab[:,fixed_index_list]==np.array(list(fixed_param_name_value_dict.values()))[np.newaxis,:]).all(axis=1))[0]
      if (len(model_index_list)==0):
         print("ERROR : Operation would result in empty grid. Aborting.")
      else :
         self.parameter_tab = self.parameter_tab[model_index_list,:] # first keep only the rows corresponding to models to be kept
         self.parameter_tab = self.parameter_tab[:,free_index_list] # then keep only the columns corresponding to free parameters
         self.observable_tab = self.observable_tab[model_index_list,:]
         self.N_models = len(model_index_list)
         self.N_parameters = len(free_index_list)
         self.parameter_names = [self.parameter_names[i] for i in free_index_list]
         self.infos += "\n# Fixed paramters:\n"
         for name in fixed_param_name_value_dict.keys() :
            self.infos += f"# {name:20} : {fixed_param_name_value_dict[name]:8.2E}\n"
            
   def reduce_parameter_range(self,param_ranges_dict):
      index_list = np.arange(self.N_models) # initialy, the indexes of all models, we will then successively eliminates the indices of models that need to be removed
      for param_name in param_ranges_dict.keys() :
         bounds = param_ranges_dict[param_name]
         param_index = self.parameter_names.index(param_name)
         if bounds[0] is not None : # if there is a lower bound, keep only the corresponding indices
            index_list = np.intersect1d(index_list , np.where(self.parameter_tab[:,param_index] >= bounds[0]))
         if bounds[1] is not None : # if there is an upper bound, keep only the corresponding indices
            index_list = np.intersect1d(index_list , np.where(self.parameter_tab[:,param_index] <= bounds[1]))
      # we now have the final list of indices of the models to keep.
      self.parameter_tab = self.parameter_tab[index_list,:]
      self.observable_tab = self.observable_tab[index_list,:]
      self.N_models = self.parameter_tab.shape[0]
      # sanitation : if one of the parameters has only one possible value left as a result, eliminate it as a fixed parameter.
      self.eliminate_fixed_parameters()
      
   def reduce_observables(self,new_observable_name_list):
      for observable_name in new_observable_name_list:
          assert observable_name in self.observable_names, "Trying to redefine the list of observables in model grid, but \"%s\" is not in the initial observable list of the model grid."%(observable_name)
      index_list = []
      for observable_name in new_observable_name_list:
          index_list.append(self.observable_names.index(observable_name))
      self.observable_names = [self.observable_names[i] for i in index_list]
      self.observable_tab = self.observable_tab[:,index_list]
      self.N_observables = len(self.observable_names)
      
   def translate_names(self,translation):
      test_forward = np.all(np.array([name in translation.forward_dict.keys() for name in self.observable_names]))
      test_reverse = np.all(np.array([name in translation.reverse_dict.keys() for name in self.observable_names]))
      assert test_forward or test_reverse, "Attempting to translate observable names but some names are not present in the translation dictionnary."
      # need to modify only : observable_names : list[str]
      new_observable_names = []
      if test_forward :
         for name in self.observable_names :
            new_observable_names.append(translation.forward_dict[name])
      else :
         for name in self.observable_names :
            new_observable_names.append(translation.reverse_dict[name])
      self.observable_names = new_observable_names

##########################
# Observation class
##########################

class Observation:
   '''add doc'''
   
   # define attributes here
   observation_id : str
   N_observables : int
   observable_names : list[str]
   observable_values : list[float]
   observable_errors : list[float]
   observable_pdfs : list[str]
   infos : str
   
   def __init__(self,observation_id,observable_names,observable_values,observable_errors,observable_pdfs=None,infos=None):
      if observable_pdfs is not None :
         observable_pdfs = np.array(observable_pdfs,dtype="str")
      assert len(observable_names)==len(observable_values), "The list of observable name should have the same number of elements as the list of observable values"
      assert (observable_pdfs is None) or np.all(np.logical_or(observable_pdfs == "normal",observable_pdfs == "lognormal")), "Error pdf type must be \"normal\" or \"lognormal\"."
      self.observation_id = observation_id
      self.N_observables = len(observable_names)
      self.observable_names = list(observable_names)
      self.observable_values = observable_values
      self.observable_errors = observable_errors
      self.observable_pdfs = observable_pdfs if observable_pdfs is not None else ["normal"]*self.N_observables
      self.infos = infos
   
   @classmethod
   def from_ASCII_file(cls,file_name,observation_id,delimiter=','):
      tab = np.loadtxt(file_name,delimiter=delimiter,dtype='str')
      if len(tab.shape)==1 : # force 2d array even if there is only one row
         tab = tab[np.newaxis,:]
      observable_names = tab[:,0].tolist()
      observable_names = list(map(str.strip, observable_names)) # remove leading and trailing blanks
      observable_values = tab[:,1].astype(float)
      observable_errors = tab[:,2].astype(float)
      if tab.shape[1] == 4 :
         observable_error_pdfs = np.array(list(map(lambda s : s.strip(),tab[:,3])))
         return cls(observation_id,observable_names,observable_values,observable_errors,observable_pdfs=observable_error_pdfs)
      else :
         return cls(observation_id,observable_names,observable_values,observable_errors)
      
   def __repr__(self):
      lines = []
      lines.append("---------------------------------------------------------")
      lines.append("Single-pointing observation                              ")
      lines.append("                                                         ")
      lines.append("ID : %s"%self.observation_id)
      lines.append(f"Number of observables : {self.N_observables:<12d}")
      lines.append("                                                         ")
      lines.append("Observable                                 Value        Error    Error_type")
      if (self.N_observables<12):
         for i in range(self.N_observables):
            lines.append('{:38s} {:12.2E} {:12.2E} {:12s}'.format(self.observable_names[i],self.observable_values[i],self.observable_errors[i],self.observable_pdfs[i]))
      else :
         for i in range(5):
            lines.append('{:38s} {:12.2E} {:12.2E} {:12s}'.format(self.observable_names[i],self.observable_values[i],self.observable_errors[i],self.observable_pdfs[i]))
         lines.append("...")
         for i in range(self.N_observables-5,self.N_observables):
            lines.append('{:38s} {:12.2E} {:12.2E} {:12s}'.format(self.observable_names[i],self.observable_values[i],self.observable_errors[i],self.observable_pdfs[i]))
      lines.append("                                                         ")
      lines.append("Additionnal infos:                                       ")
      lines.append(self.infos if self.infos is not None else "None")
      lines.append("---------------------------------------------------------")
      return "\n".join(lines)
   
   def summary(self):
      print(self)
      
   def copy(self):
      return copy.deepcopy(self)
      
   def get_observation_id(self):
      return self.observation_id
   
   def get_N_observables(self):
      return self.N_observables
   
   def get_observable_names(self):
      return self.observable_names.copy()
      
   def get_observable_vector(self):
      return np.array(self.observable_values).copy()
      
   def get_error_vector(self):
      return np.array(self.observable_errors).copy()
      
   def get_error_type_vector(self):
      return np.array(self.observable_pdfs.copy(),dtype='str')
      
   def get_observable_value(self,observable_name):
      assert observable_name in self.observable_names, "Trying to access observable value in Observationfor observable name not present in Observation."
      index_param = self.observable_names.index(observable_name)
      return self.observable_values[index_param]
      
   def get_observable_error(self,observable_name):
      assert observable_name in self.observable_names, "Trying to access observable value in Observationfor observable name not present in Observation."
      index_param = self.observable_names.index(observable_name)
      return self.observable_errors[index_param]
      
   def get_observable_error_pdf(self,observable_name):
      assert observable_name in self.observable_names, "Trying to access observable value in Observationfor observable name not present in Observation."
      index_param = self.observable_names.index(observable_name)
      return self.observable_pdfs[index_param]
      
   def reduce_observables(self,new_observable_name_list):
      for observable_name in new_observable_name_list:
          assert observable_name in self.observable_names, "Trying to redefine the list of observables in observation, but \"%s\" is not in the initial observable list of the observation."%(observable_name)
      index_list = []
      for observable_name in new_observable_name_list:
          index_list.append(self.observable_names.index(observable_name))
      self.observable_names = [self.observable_names[i] for i in index_list]
      self.observable_values = [self.observable_values[i] for i in index_list]
      self.observable_errors = [self.observable_errors[i] for i in index_list]
      self.observable_pdfs = [self.observable_pdfs[i] for i in index_list]
      self.N_observables = len(self.observable_names)
      
   def expand_reduce_observables(self,new_observable_name_list):
      index_list = []
      for observable_name in new_observable_name_list :
         if observable_name in self.observable_names :
            index_list.append(self.observable_names.index(observable_name))
         else :
            index_list.append(-1)
      self.observable_names = new_observable_name_list
      self.observable_values = [self.observable_values[i] if i>=0 else np.nan for i in index_list]
      self.observable_errors = [self.observable_errors[i] if i>=0 else np.nan for i in index_list]
      self.observable_pdfs = [self.observable_pdfs[i] if i>=0 else "normal" for i in index_list]
      self.N_observables = len(self.observable_names)
      
   def translate_names(self,translation):
      test_forward = np.all(np.array([name in translation.forward_dict.keys() for name in self.observable_names]))
      test_reverse = np.all(np.array([name in translation.reverse_dict.keys() for name in self.observable_names]))
      assert test_forward or test_reverse, "Attempting to translate observable names but some names are not present in the translation dictionnary."
      # need to modify only : observable_names : list[str]
      new_observable_names = []
      if test_forward :
         for name in self.observable_names :
            new_observable_names.append(translation.forward_dict[name])
      else :
         for name in self.observable_names :
            new_observable_names.append(translation.reverse_dict[name])
      self.observable_names = new_observable_names
      
   def set_multiplicative_noise_threshold(self,threshold=0.2,style="true_mult"):
      # Turn the lsits into array
      obs_vector = np.array(self.observable_values)
      obs_err_vector = np.array(self.observable_errors)
      obs_err_type = np.array(self.observable_pdfs,dtype='object')
      # Modify error bars
      if style == "true_mult" :
         mask = (obs_err_vector/obs_vector < threshold) & (obs_vector > 0.) # forbid switching to lognormal error type if obs value is negative !
         obs_err_type[mask] = "lognormal"
         obs_err_vector[mask] = 1. + threshold
      elif style == "old_additive_approx" :
         mask = (obs_err_vector/obs_vector < threshold) & (obs_vector > 0.)
         obs_err_vector[mask] = (threshold * obs_vector)[mask]
      else :
         print("Warning : \"style\" must be either \"true_mult\" or \"old_additive_approx\". No effect otherwise.")
      # Put the modified arrays in the object's attribute
      self.observable_values = list(obs_vector)
      self.observable_errors = list(obs_err_vector)
      self.observable_pdfs   = list(obs_err_type)
    
##########################
# Observation_map class
##########################

class Observation_map:
   '''add doc'''
   
   # define attributes here
   map_id : str
   N_observables : int
   N_pixels : int
   observable_names : list[str]
   observation_list : list[Observation]
   pixel_index_list : list
   infos : str
   
   def __init__(self,map_id,observable_names,observation_list,pixel_index_list,infos=None):
      N_observables = observation_list[0].get_N_observables()
      for obs in observation_list :
         assert obs.get_N_observables() == N_observables, "Trying to create an observation map with pixels that do not have the same number of observables."
      observable_names = observation_list[0].get_observable_names()
      for obs in observation_list :
         assert obs.get_observable_names() == observable_names, "Trying to create an observation map with pixels that do not have the same list of observable names."
      assert (len(observation_list)==len(pixel_index_list)), "Trying to create an observation map but the list of observation objects and the list of pixel indices do not have the same length."
      #error_pdf = observation_list[0].get_error_type_vector()
      #for obs in observation_list :
      #  assert obs.get_error_type_list() == error_pdf, "When creating an Observation_map object, all pixels must have the same error PDFs."
      assert all( ((type(index) is tuple) and len(index)==2) for index in pixel_index_list ), "The index list for creating an Observation_map object must be a list of tuples, each containing the two indices of a given pixel."
      
      self.map_id = map_id
      self.N_observables = N_observables
      self.N_pixels = len(observation_list)
      self.observable_names = observable_names
      self.observation_list = observation_list
      self.pixel_index_list = pixel_index_list
      self.infos = infos
      
   @classmethod
   def from_ASCII_file(cls,obs_file_name,err_file_name,map_id):
      ## Reading the obs file
      # First read number of observables and list of observable names
      obs_file = open(obs_file_name)
      line_counter = 0
      line = obs_file.readline()
      line_counter += 1
      while(line[0] == '#'):
         line = obs_file.readline()
         line_counter += 1
      N_observables = int(line)
      line = obs_file.readline()
      line_counter += 1
      while(line[0] == '#'):
         line = obs_file.readline()
         line_counter += 1
      observable_names = []
      for i in range(N_observables):
         observable_names.append(line.split("|")[0].strip())
         line = obs_file.readline()
         line_counter += 1
      obs_file.close()
      assert len(observable_names) == N_observables, "The number of observable names listed at the beginning of the observable file is not consistent with the announced number of observables."
      # Second read the data array
      obs_tab = np.loadtxt(obs_file_name,skiprows=line_counter)
      N_pixels = obs_tab.shape[0]
      assert obs_tab.shape[1] == N_observables+2, "The number of columns in the observable file is not consistent with the announced number of observables."
      ## Reading the err file
      # First read number of observables and list of observable names (check that it is identical to the err file)
      err_file = open(err_file_name)
      line_counter = 0
      line = err_file.readline()
      line_counter += 1
      while(line[0] == '#'):
         line = err_file.readline()
         line_counter += 1
      N_observables2 = int(line)
      assert N_observables==N_observables2, "When reading an observation map from files, the number of observables in the observable file and in the error file must be the same."
      line = err_file.readline()
      line_counter += 1
      while(line[0] == '#'):
         line = err_file.readline()
         line_counter += 1
      observable_names2 = []
      for i in range(N_observables):
         observable_names2.append(line.split("|")[0].strip())
         line = err_file.readline()
         line_counter += 1
      err_file.close()
      assert N_observables==N_observables2, "When reading an observation map from files, the list of observable names in the observable file and in the error file must be the identical."
      # Second read the data array
      err_tab = np.loadtxt(err_file_name,skiprows=line_counter)
      assert obs_tab.shape[0] == N_pixels, "The number of pixel in the error file is not consistent with the number of pixels in the observable file."
      assert obs_tab.shape[1] == N_observables+2, "The number of columns in the error file is not consistent with the announced number of observables."
      ## Produce the index list
      pixel_index_list = [ tuple(obs_tab[i,:2].astype(int)) for i in range(N_pixels)]
      pixel_index_list2 = [ tuple(err_tab[i,:2].astype(int)) for i in range(N_pixels)]
      assert pixel_index_list == pixel_index_list2, "Pixels (and their indices) should be the same (and in the same order) in the observable file and in the error file."
      ## Build the list of observations
      pixel_index_list = [] # rebuild pixel index list because we want to exclude pixels with no detected line
      observation_list = []
      for i in range(N_pixels):
         if not np.all(np.isnan(obs_tab[i,2:])) :
            pixel_index_list.append(tuple(obs_tab[i,:2].astype(int)))
            observation_list.append(Observation(map_id+" (pixel %i %i)"%(pixel_index_list[i][0],pixel_index_list[i][1]),observable_names,obs_tab[i,2:],err_tab[i,2:]))
      
      ## return the resulting Observation_map object
      return cls(map_id,observable_names,observation_list,pixel_index_list)
      
   def __repr__(self):
      lines = []
      lines.append("---------------------------------------------------------")
      lines.append("Observation map                                          ")
      lines.append("                                                         ")
      lines.append("ID : %s"%self.map_id)
      lines.append(f"Number of pixels : {self.N_pixels:<6d} number of observables : {self.N_observables:<6d}")
      lines.append("                                                         ")
      lines.append("List of Observable                      min value    max value    min_error    max_error    Error_type")
      if (self.N_observables<12):
         for i in range(self.N_observables):
            obs_values = self.get_single_observable_map(self.observable_names[i])
            err_values = self.get_single_observable_error_map(self.observable_names[i])
            err_pdf = self.get_single_observable_error_pdfs(self.observable_names[i])
            if len(set(err_pdf)) == 1 :
               err_pdf = err_pdf[0]
            else :
               err_pdf = "mult. types"
            lines.append('{:36s} {:12.2E} {:12.2E} {:12.2E} {:12.2E}     {:12s}'.format(self.observable_names[i],np.nanmin(obs_values),np.nanmax(obs_values),np.nanmin(err_values),np.nanmax(err_values),err_pdf))
      else :
         for i in range(5):
            obs_values = self.get_single_observable_map(self.observable_names[i])
            err_values = self.get_single_observable_error_map(self.observable_names[i])
            err_pdf = self.get_single_observable_error_pdfs(self.observable_names[i])
            if len(set(err_pdf)) == 1 :
               err_pdf = err_pdf[0]
            else :
               err_pdf = "mult. types"
            lines.append('{:36s} {:12.2E} {:12.2E} {:12.2E} {:12.2E}     {:12s}'.format(self.observable_names[i],np.nanmin(obs_values),np.nanmax(obs_values),np.nanmin(err_values),np.nanmax(err_values),err_pdf))
         lines.append("...")
         for i in range(self.N_observables-5,self.N_observables):
            obs_values = self.get_single_observable_map(self.observable_names[i])
            err_values = self.get_single_observable_error_map(self.observable_names[i])
            err_pdf = self.get_single_observable_error_pdfs(self.observable_names[i])
            if len(set(err_pdf)) == 1 :
               err_pdf = err_pdf[0]
            else :
               err_pdf = "mult. types"
            lines.append('{:36s} {:12.2E} {:12.2E} {:12.2E} {:12.2E}     {:12s}'.format(self.observable_names[i],min(obs_values),max(obs_values),min(err_values),max(err_values),err_pdf))
      lines.append("                                                         ")
      lines.append("Additionnal infos:                                       ")
      lines.append(self.infos if self.infos is not None else "None")
      lines.append("---------------------------------------------------------")
      return "\n".join(lines)
      
   def summary(self):
      print(self)
      
   def copy(self):
      return copy.deepcopy(self)
      
   def get_map_id(self):
      return self.map_id
      
   def get_N_observables(self):
      return self.N_observables
      
   def get_N_pixels(self):
      return self.N_pixels
      
   def get_observable_names(self):
      return self.observable_names
      
   def get_observation_list(self):
      return self.observation_list
      
   def get_pixel_index_list(self):
      return self.pixel_index_list
      
   def get_infos(self):
      return self.infos
      
   def get_single_pixel_observation(self,index):
      return self.observation_list[index]
      
   def get_single_pixel_indices(self,index):
      return self.pixel_index_list[index]
      
   def get_single_observable_map(self,observable_name):
      assert observable_name in self.observable_names, "Trying to access single observable map in Observation_map for observable name not present in this Observation_map."
      return [observation.get_observable_value(observable_name) for observation in self.observation_list]
      
   def get_single_observable_error_map(self,observable_name):
      assert observable_name in self.observable_names, "Trying to access single observable map in Observation_map for observable name not present in this Observation_map."
      return [observation.get_observable_error(observable_name) for observation in self.observation_list]
      
   def get_single_observable_error_pdfs(self,observable_name):
      assert observable_name in self.observable_names, "Trying to access single observable map in Observation_map for observable name not present in this Observation_map."
      return [observation.get_observable_error_pdf(observable_name) for observation in self.observation_list]
      
   def reduce_observables(self,new_observable_name_list):
      for observable_name in new_observable_name_list:
          assert observable_name in self.observable_names, "Trying to redefine the list of observables in observation, but \"%s\" is not in the initial observable list of the observation."%(observable_name)
      index_list = []
      for observable_name in new_observable_name_list:
          index_list.append(self.observable_names.index(observable_name))
      self.observable_names = [self.observable_names[i] for i in index_list]
      for observation in self.observation_list :
         observation.reduce_observables(new_observable_name_list)
      self.N_observables = len(self.observable_names)
      
   def expand_reduce_observables(self,new_observable_name_list):
      index_list = []
      for observable_name in new_observable_name_list :
         if observable_name in self.observable_names :
            index_list.append(self.observable_names.index(observable_name))
         else :
            index_list.append(-1)
      self.observable_names = new_observable_name_list
      for observation in self.observation_list :
         observation.expand_reduce_observables(new_observable_name_list)
      self.N_observables = len(self.observable_names)
   
   def translate_names(self,translation):
      test_forward = np.all(np.array([name in translation.forward_dict.keys() for name in self.observable_names]))
      test_reverse = np.all(np.array([name in translation.reverse_dict.keys() for name in self.observable_names]))
      assert test_forward or test_reverse, "Attempting to translate observable names but some names are not present in the translation dictionnary."
      # need to modify only : observable_names : list[str]
      new_observable_names = []
      if test_forward :
         for name in self.observable_names :
            new_observable_names.append(translation.forward_dict[name])
      else :
         for name in self.observable_names :
            new_observable_names.append(translation.reverse_dict[name])
      self.observable_names = new_observable_names
      for observation in self.observation_list :
         observation.translate_names(translation)
         
   def set_multiplicative_noise_threshold(self,threshold=0.2,style="true_mult"):
      for observation in self.observation_list :
         observation.set_multiplicative_noise_threshold(threshold=threshold,style=style)

##########################
# Translation class
##########################

class Translation:
   '''add doc'''
   
   # define attributes here
   name : str
   forward_dict : dict
   reverse_dict : dict
   N_pairs : int
   
   def __init__(self,name,forward_dict):
      self.name = name
      self.forward_dict = forward_dict
      self.reverse_dict = {value:key for key,value in forward_dict.items()}
      self.N_pairs = len(forward_dict.keys())
     
   @classmethod
   def from_ASCII_file(cls,file_name,name=None,delimiter='|'):
      tab = np.loadtxt(file_name,delimiter=delimiter,dtype='str')
      if len(tab.shape)==1 : # if single line in file
         tab = tab[np.newaxis,:]
      my_dict = {}
      for i in range(tab.shape[0]):
         my_dict[tab[i,0].strip()] = tab[i,1].strip()
      forward_dict = my_dict
      if name is None :
         name = '.'.join((file_name.split('/')[-1]).split('.')[:-1])
      return cls(name,forward_dict)
      
   def __repr__(self):
      lines = []
      lines.append("---------------------------------------------------------")
      lines.append("Name translation                                         ")
      lines.append("                                                         ")
      lines.append(f"Name : {self.name:s}                                        ")
      lines.append(f"Number of name pairs: {len(self.forward_dict.keys()):<12d}  ")
      lines.append("Name               Translation")
      dict_keys = list(self.forward_dict.keys())
      if (self.N_pairs <12):
         for i in range(self.N_pairs):
            key = dict_keys[i]
            lines.append('{:20s} {:20s}'.format(key,self.forward_dict[key]))
      else :
         for i in range(5):
            key = dict_keys[i]
            lines.append('{:20s} {:20s}'.format(key,self.forward_dict[key]))
         lines.append("...")
         for i in range(self.N_pairs-5,self.N_pairs):
            key = dict_keys[i]
            lines.append('{:20s} {:20s}'.format(key,self.forward_dict[key]))
      lines.append("---------------------------------------------------------")
      return "\n".join(lines)
   
   def summary(self):
      print(self)
      
   def copy(self):
      return copy.deepcopy(self)
      
   def get_N_pairs(self):
      return self.N_pairs
      
   def get_name(self):
      return self.name
      
   def get_forward_dict(self):
      return self.forward_dict.copy()
      
   def get_reverse_dict(self):
      return self.reverse_dict.copy()
      
   def get_names(self):
      return list(self.forward_dict.keys())
   
   def get_alternate_names(self):
      return list(self.reverse_dict.keys())
      
##########################
# Grid_approximator class
##########################

class Grid_approximator:
   '''add doc'''
   
   # define attributes here
   N_parameters : int
   N_observables : int
   parameter_names : list[str]
   parameter_bounds : np.ndarray
   observable_names : list[str]
   interpolator_type : str
   interpolators : list
   code_version : str
   grid_id : str
   
   def __init__(self):
      self.N_parameters = None
      self.N_observables = None
      self.parameter_names = None
      self.parameter_bounds = None
      self.observable_names = None
      self.interpolator_type = None
      self.interpolators = None
      self.code_version = None
      self.grid_id = None
      self.grid_parameters = None # Used only for ISMFIT interpolation
   
   @classmethod
   def interpolate_from_grid(cls,grid,interpolator_type="LogRBF",**kwargs):
      assert interpolator_type in ["Linear","LogLinear","RBF","LogRBF","ISMFIT"], "%s is not a possible choice of interpolator type."%interpolator_type
      assert interpolator_type != "RBF", "RBF interpolation (in linear scale) performs very poorly, please use \"LogRBF\" (RBF interpolation in log-log) instead."

      # Set or receive RBF kernel type in the case of RBF interpolation
      if "RBF_kernel" in kwargs:
         RBF_kernel = kwargs["RBF_kernel"]
         if interpolator_type == "RBF" and RBF_kernel != "Linear" :
            print("Warning : \"RBF\" interpolator type can only uses a linear kernel. Reverting to linear kernel.")
            RBF_kernel = "linear"
         elif interpolator_type != "LogRBF" :
            print("Warning : optional argument \"RBF_kernel\" not used if interpolator type different from \"RBF\" or \"LogRBF\".")
      else :
         if interpolator_type == "RBF" :
            RBF_kernel = "linear"
         elif interpolator_type == "LogRBF" :
            RBF_kernel = "quintic"
      
      # Build Grid_approximator object and fill the simple attributes
      approximator = Grid_approximator()
      approximator.N_parameters = grid.N_parameters
      approximator.N_observables = grid.N_observables
      approximator.parameter_names = grid.parameter_names
      approximator.observable_names = grid.observable_names
      approximator.code_version = grid.code_version
      approximator.grid_id = grid.grid_id
      grid_parameters = grid.get_parameter_array()
      param_maxs = grid_parameters.max(axis=0)
      param_mins = grid_parameters.min(axis=0)
      approximator.parameter_bounds = np.vstack((param_mins,param_maxs))
      
      # Get the data from the grid
      grid_parameters = grid.get_parameter_array()
      grid_observables = grid.get_observable_array()
         
      # Build the interpolators
      if interpolator_type == "Linear":
         if grid.N_parameters > 1 :
            approximator.interpolators = [scipy.interpolate.LinearNDInterpolator(grid_parameters,grid_observables[:,i],rescale=True) for i in tqdm(range(approximator.N_observables))]
         else : # special treatment for 1D grid case
            approximator.interpolators = [scipy.interpolate.interp1d(grid_parameters[:,0],grid_observables[:,i],kind='linear') for i in tqdm(range(approximator.N_observables))]
      elif interpolator_type == "LogLinear":
         if grid.N_parameters > 1 :
            approximator.interpolators = [scipy.interpolate.LinearNDInterpolator(np.log10(grid_parameters),np.log10(grid_observables[:,i]),rescale=True) for i in tqdm(range(approximator.N_observables))]
         else : # special treatment for 1D grid case
            approximator.interpolators = [scipy.interpolate.interp1d(np.log10(grid_parameters)[:,0],np.log10(grid_observables[:,i]), kind='linear') for i in tqdm(range(approximator.N_observables))]
      elif interpolator_type == "RBF":
         approximator.interpolators = [scipy.interpolate.RBFInterpolator(grid_parameters,grid_observables[:,i],kernel=RBF_kernel) for i in tqdm(range(approximator.N_observables))]
      elif interpolator_type == "LogRBF":
         approximator.interpolators = [scipy.interpolate.RBFInterpolator(np.log10(grid_parameters),np.log10(grid_observables[:,i]),kernel=RBF_kernel) for i in tqdm(range(approximator.N_observables))]
      elif interpolator_type == "ISMFIT":
         approximator.interpolators = pyismtools.utils.build_ISMFIT_interpolators(grid_parameters,grid_observables)
         approximator.grid_parameters = grid_parameters
         
      # store interpolators in the correct attribute
      approximator.interpolator_type = interpolator_type
      
      # return the object
      return approximator
         
   def __call__(self,parameter_vector,allow_extrapolation=False):
      eps = 1E-9 # tolerance on calls outside the bounds
      parameter_vector = np.array(parameter_vector)
      if allow_extrapolation :
         # if we are extrapolating, we force constant extrapolation
         effective_parameter_vector = np.maximum(parameter_vector,self.get_parameter_mins())
         effective_parameter_vector = np.minimum(parameter_vector,self.get_parameter_maxs())
      else:
         # if not allowed, check that we are not extrapolating
         if np.any(parameter_vector > self.get_parameter_maxs()*(1. + eps)) or np.any(parameter_vector < self.get_parameter_mins()*(1. - eps)):
            raise ValueError("Interpolator trying to extrapolate : parameter vector ["+','.join([" %.20E"%(x) for x in list(parameter_vector)])+"] while min_bounds were ["+','.join([" %.20E"%(x) for x in list(self.get_parameter_mins())])+"] and max_bounds were ["+','.join([" %.20E"%(x) for x in list(self.get_parameter_maxs())])+"]")
         effective_parameter_vector = np.maximum(parameter_vector,self.get_parameter_mins()) # We still do constant extrapolation if out of bounds but within our tolerance
         effective_parameter_vector = np.minimum(parameter_vector,self.get_parameter_maxs()) #
      if self.interpolator_type == "Linear":
         observable_vector = np.array([self.interpolators[i](effective_parameter_vector)[0] for i in range(self.N_observables)])
      elif self.interpolator_type == "LogLinear":
         observable_vector = np.array([10**self.interpolators[i](np.log10(effective_parameter_vector))[0] for i in range(self.N_observables)])
      elif self.interpolator_type == "RBF":
         observable_vector = np.array([self.interpolators[i](effective_parameter_vector[np.newaxis,:])[0] for i in range(self.N_observables)])
      elif self.interpolator_type == "LogRBF":
         observable_vector = np.array([10**self.interpolators[i](np.log10(effective_parameter_vector)[np.newaxis,:])[0] for i in range(self.N_observables)])
      elif self.interpolator_type == "ISMFIT":
         observable_vector = pyismtools.utils.call_ISMFIT_interpolators(effective_parameter_vector,self.grid_parameters,self.interpolators)
         
      interpolated_model = Model(self.parameter_names.copy(),list(parameter_vector),self.observable_names.copy(),list(observable_vector),self.code_version,self.grid_id,interpolated_model=True)
      return interpolated_model
      
   def vectorized_call(self,parameter_matrix):
   #   # the parameter matrix should contain as rows the multiple parameter vectors for which the interpolation must be computed
      if self.interpolator_type == "Linear":
         observable_matrix = np.fromiter(map(methodcaller('__call__',parameter_matrix), self.interpolators),dtype=np.dtype((float,parameter_matrix.shape[0]))).T
      elif self.interpolator_type == "LogLinear":
         observable_matrix = 10**np.fromiter(map(methodcaller('__call__',np.log10(parameter_matrix)), self.interpolators),dtype=np.dtype((float,parameter_matrix.shape[0]))).T
      elif self.interpolator_type == "RBF":
         observable_matrix = np.fromiter(map(methodcaller('__call__', parameter_matrix), self.interpolators),dtype=np.dtype((float,parameter_matrix.shape[0]))).T
      elif self.interpolator_type == "LogRBF":
         observable_matrix = 10**np.fromiter(map(methodcaller('__call__',np.log10(parameter_matrix)), self.interpolators),dtype=np.dtype((float,parameter_matrix.shape[0]))).T
      elif self.interpolator_type == "ISMFIT":
         sys.exit('Method vectorized_call is not implemented yet for approximator type "ISMFIT"')
         #observable_matrix = pyismtools.utils.call_ISMFIT_interpolators(parameter_matrix,self.grid_parameters,self.interpolators)
      return observable_matrix
      
   def __repr__(self):
      lines = []
      lines.append("---------------------------------------------------------")
      lines.append("Grid approximator                                        ")
      lines.append(f"Grid ID : {self.grid_id:21}  code version : {self.code_version:12}")
      lines.append(f"interpolator type : {self.interpolator_type:12}")
      lines.append(f"Number of parameters : {self.N_parameters:<6d} number of observables : {self.N_observables:<6d}")
      lines.append("                                                         ")
      lines.append("Parameters :                                             ")
      lines.append("name                             min_value  max_value")
      for i in range(self.N_parameters):
         lines.append(f"{self.parameter_names[i]:30} {self.parameter_bounds[0,i]:10.2E} {self.parameter_bounds[1,i]:11.2E}")
      lines.append("                                                         ")
      lines.append("Observables :                                            ")
      lines.append("name                                                     ")
      if (self.N_observables<12):
         for i in range(self.N_observables):
            lines.append(f"{self.observable_names[i]:30}")
      else :
         for i in range(5):
            lines.append(f"{self.observable_names[i]:30}")
         lines.append("...")
         for i in range(self.N_observables-5,self.N_observables):
            lines.append(f"{self.observable_names[i]:30}")
      lines.append("---------------------------------------------------------")
      return "\n".join(lines)
      
   def copy(self):
      return copy.deepcopy(self)
      
   def get_N_parameters(self):
      return self.N_parameters
      
   def get_N_observables(self):
      return self.N_observables
      
   def get_code_version(self):
      return self.code_version
   
   def get_grid_id(self):
      return self.grid_id
      
   def get_interpolator_type(self):
      return self.interpolator_type
      
   def get_parameter_names(self):
      return self.parameter_names.copy()
      
   def get_observable_names(self):
      return self.observable_names.copy()
      
   def get_parameter_maxs(self):
      return self.parameter_bounds[1,:]
      
   def get_parameter_mins(self):
      return self.parameter_bounds[0,:]
      
   def reduce_observables(self,new_observable_name_list):
      for observable_name in new_observable_name_list:
          assert observable_name in self.observable_names, "Trying to redefine the list of observables in Grid_approximator, but \"%s\" is not in the initial observable list of the Grid_approximator."%(observable_name)
      index_list = []
      for observable_name in new_observable_name_list:
          index_list.append(self.observable_names.index(observable_name))
      self.observable_names = [self.observable_names[i] for i in index_list]
      self.interpolators = [self.interpolators[i] for i in index_list]
      self.N_observables = len(self.observable_names)
