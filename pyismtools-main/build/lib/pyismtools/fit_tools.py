from pyismtools.classes import *
import pyismtools.utils
import scipy.optimize
from tqdm.auto import tqdm


def compute_cost_function_model(my_observation,my_model,cost_function="chi2"):

   ## 1 - reduce model and observation to set of common observable quantities
   
   # copy objects to avoid modifying them
   my_observation = my_observation.copy()
   my_model = my_model.copy()
   
   # reduce observables to the common set between model and observations
   obs_name_set = set(my_observation.get_observable_names())
   model_name_set = set(my_model.get_observable_names())
   
   intersection_set = list( model_name_set.intersection(obs_name_set) )
   assert len(intersection_set)>0, "Trying to compare a Model and an Observation that have no common observable quantities"
   
   my_observation.reduce_observables(intersection_set)
   my_model.reduce_observables(intersection_set)

   ## 2 - Perform cost function computation
      
   # translate string for cost function if given
   if cost_function == "chi2":
      cost_function = pyismtools.utils.chi2
   elif cost_function == "reduced_chi2":
      cost_function = pyismtools.utils.reduced_chi2
   elif cost_function == "chi2_ISMFIT":
      cost_function = pyismtools.utils.chi2_ISMFIT

   # compute cost function of model
   return cost_function(my_model.get_scaling_factor()*my_model.get_observable_vector(scaled=False),my_observation.get_observable_vector(),my_observation.get_error_vector(),my_observation.get_error_type_vector(),my_model.get_N_parameters())


def find_best_model_in_grid(observation_or_map,my_model_grid,cost_function="chi2",with_scaling_factor=True,scaling_factor_bounds=[0.1,10.],initial_scaling_factor = 1):
   if isinstance(observation_or_map,Observation) :
      return find_best_model_in_grid_single_obs(observation_or_map,my_model_grid,cost_function=cost_function,with_scaling_factor=with_scaling_factor,scaling_factor_bounds=scaling_factor_bounds,initial_scaling_factor=initial_scaling_factor)
   elif isinstance(observation_or_map,Observation_map) :
      return find_best_model_in_grid_map(observation_or_map,my_model_grid,cost_function=cost_function,with_scaling_factor=with_scaling_factor,scaling_factor_bounds=scaling_factor_bounds,initial_scaling_factor=initial_scaling_factor)
   else :
      raise ValueError("The first argument to function find_best_model_in_grid should be either of type Observation or of type Observation_map")


def find_best_model_in_grid_single_obs(my_observation,my_model_grid,cost_function="chi2",with_scaling_factor=True,scaling_factor_bounds=[0.1,10.],initial_scaling_factor = 1):

   ## 1 - reduce model grid and observation to set of common observable quantities
   
   # work with copies of objects to avoid modifying them
   my_observation_cp = my_observation.copy()
   my_model_grid_cp = my_model_grid.copy()
   
   # reduce observables to the common set between model and observations
   obs_name_set = set(my_observation_cp.get_observable_names())
   model_name_set = set(my_model_grid_cp.get_observable_names())
   
   intersection_set = list( model_name_set.intersection(obs_name_set) )
   assert len(intersection_set)>0, "Trying to compare a Model_grid and an Observation that have no common observable quantities"
   
   my_observation_cp.reduce_observables(intersection_set)
   my_model_grid_cp.reduce_observables(intersection_set)
   
   ## 2 - Find best model in grid
      
   # translate string for cost function if given
   if cost_function == "chi2":
      cost_function = pyismtools.utils.chi2
   elif cost_function == "reduced_chi2":
      cost_function = pyismtools.utils.reduced_chi2
   elif cost_function == "chi2_ISMFIT":
      cost_function = pyismtools.utils.chi2_ISMFIT
      
   # get grid data
   grid_observable_array = my_model_grid_cp.get_observable_array()
   grid_param_array = my_model_grid_cp.get_parameter_array()
   best_scaling_factors = np.ones(grid_param_array.shape[0])
   
   # find best fitting scaling factor for each model of the grid
   if with_scaling_factor==True:
      def temp_function(scaling_factor,model_obs_vector):
         return cost_function(scaling_factor*model_obs_vector,my_observation_cp.get_observable_vector(),my_observation_cp.get_error_vector(),my_observation_cp.get_error_type_vector(),my_model_grid_cp.get_N_parameters())
      for i in range(grid_param_array.shape[0]):
         temp_res = scipy.optimize.minimize_scalar(temp_function,args=(grid_observable_array[i,:][np.newaxis,:]),bracket=[scaling_factor_bounds[0],initial_scaling_factor,scaling_factor_bounds[1]],bounds=scaling_factor_bounds,method="Bounded")
         best_scaling_factors[i] = temp_res.x
         if not temp_res.success :
            print("Model %i ("%(i),grid_param_array[i,:],") : %s. Try improving your initial guess for the scaling factor"%(temp_res.message),temp_res.x)
   
   # compute cost function for all models in grid
   cost_array = cost_function(grid_observable_array*best_scaling_factors[:,np.newaxis],my_observation_cp.get_observable_vector(),my_observation_cp.get_error_vector(),my_observation_cp.get_error_type_vector(),my_model_grid_cp.get_N_parameters())
   
   # return best model
   index_min = cost_array.argmin()
   cost_min = cost_array.min()
   best_model = my_model_grid.get_model(grid_param_array[index_min,:]) # take from the original grid to have all observables
   if with_scaling_factor==True :
      best_model.set_scaling_factor(best_scaling_factors[index_min])
   return best_model, cost_min


def find_best_model_in_grid_map(my_observation_map,my_model_grid,cost_function="chi2",with_scaling_factor=True,scaling_factor_bounds=[0.1,10.],initial_scaling_factor = 1):
   best_model_list = []
   best_chi2_list = []
   for observation in my_observation_map.get_observation_list() :
      best_model, best_chi2 = find_best_model_in_grid_single_obs(observation,my_model_grid,cost_function=cost_function,with_scaling_factor=with_scaling_factor,scaling_factor_bounds=scaling_factor_bounds,initial_scaling_factor=initial_scaling_factor)
      best_model_list.append(best_model)
      best_chi2_list.append(best_chi2)
   return best_model_list, best_chi2_list


def find_best_scaling_factor_single_model(my_observation,my_model,cost_function="chi2",initial_scaling_factor=1.,scaling_factor_bounds=[0.1,10.],return_model = True):
   
   ## 1 - reduce model and observation to set of common observable quantities
   
   # work with copies of objects to avoid modifying them
   my_observation_cp = my_observation.copy()
   my_model_cp = my_model.copy()
   
   # reduce observables to the common set between model and observations
   obs_name_set = set(my_observation_cp.get_observable_names())
   model_name_set = set(my_model_cp.get_observable_names())
   
   intersection_set = list( model_name_set.intersection(obs_name_set) )
   assert len(intersection_set)>0, "Trying to compare a Model and an Observation that have no common observable quantities"
   
   my_observation_cp.reduce_observables(intersection_set)
   my_model_cp.reduce_observables(intersection_set)
   
   ## 2 - find best scaling factor for model
   
   # translate string for cost function if given
   if cost_function == "chi2":
      cost_function = pyismtools.utils.chi2
   elif cost_function == "reduced_chi2":
      cost_function = pyismtools.utils.reduced_chi2
   elif cost_function == "chi2_ISMFIT":
      cost_function = pyismtools.utils.chi2_ISMFIT
      
   model_obs_vector = my_model_cp.get_observable_vector()
      
   def temp_function(scaling_factor):
         return cost_function(scaling_factor*model_obs_vector,my_observation_cp.get_observable_vector(),my_observation_cp.get_error_vector(),my_observation_cp.get_error_type_vector(),my_model_cp.get_N_parameters())
         
   temp_res = scipy.optimize.minimize_scalar(temp_function,bracket=[scaling_factor_bounds[0],initial_scaling_factor,scaling_factor_bounds[1]],bounds=scaling_factor_bounds,method="Bounded")
   if not temp_res.success :
            print("Model %i ("%(i),grid_param_array[i,:],") : %s. Try improving your initial guess for the initial guess."%(temp_res.message),temp_res.x)
   scaling_factor = temp_res.x#[0] if len(model_obs_vector)>1 else temp_res.x
   
   if return_model == True :
      scaled_model = my_model.copy() # return the full model rather than the reduced one
      scaled_model.set_scaling_factor(scaling_factor)
      return scaled_model
   else :
      return scaling_factor
      
def find_best_interpolated_model(observation_or_map,my_approximator,initial_parameters=None,cost_function="chi2",with_scaling_factor=True,initial_scaling_factor=1.,scaling_factor_bounds=[0.1,10.],method="SLSQP",N_repetitions = 1):
   if isinstance(observation_or_map,Observation) :
      return find_best_interpolated_model_single_obs(observation_or_map,my_approximator,initial_parameters=initial_parameters,cost_function=cost_function,with_scaling_factor=with_scaling_factor,initial_scaling_factor=initial_scaling_factor,scaling_factor_bounds=scaling_factor_bounds,method=method,N_repetitions=N_repetitions)
   elif isinstance(observation_or_map,Observation_map) :
      return find_best_interpolated_model_map(observation_or_map,my_approximator,initial_parameters=initial_parameters,cost_function=cost_function,with_scaling_factor=with_scaling_factor,initial_scaling_factor=initial_scaling_factor,scaling_factor_bounds=scaling_factor_bounds,method=method,N_repetitions=N_repetitions)
   else :
      raise ValueError("The first argument to function find_best_interpolated_model should be either of type Observation or of type Observation_map")

def find_best_interpolated_model_single_obs(my_observation,my_approximator,initial_parameters=None,cost_function="chi2",with_scaling_factor=True,initial_scaling_factor=1.,scaling_factor_bounds=[0.1,10.],method="SLSQP",N_repetitions = 1,quiet=False):
   
   ## 1 - reduce approximator and observation to set of common observable quantities
   
   # work with copies of objects to avoid modifying them
   my_observation_cp = my_observation.copy()
   my_approximator_cp = my_approximator.copy()
   
   # reduce observables to the common set between model and observations
   obs_name_set = set(my_observation_cp.get_observable_names())
   model_name_set = set(my_approximator_cp.get_observable_names())
   
   intersection_list = list( model_name_set.intersection(obs_name_set) )
   assert len(intersection_list)>0, "Trying to compare a Grid_approximator and an Observation that have no common observable quantities"
   assert method not in ["Newton-CG","dogleg","trust-ncg","trust-exact","trust-krylov","COBYLA","trust-constr"], "Method %s is not supported by pyismtools (supported methods are ‘Nelder-Mead’,‘Powell’, ‘TNC’, ‘L-BFGS-B’, ‘SLSQP’, ‘BFGS’, ‘CG’)."%(method)
   
   my_observation_cp.reduce_observables(intersection_list)
   my_approximator_cp.reduce_observables(intersection_list)
   
   ## 2 - find best interpolated model
      
   # translate string for cost function if given
   if cost_function == "chi2":
      cost_function = pyismtools.utils.chi2
   elif cost_function == "reduced_chi2":
      cost_function = pyismtools.utils.reduced_chi2
   elif cost_function == "chi2_ISMFIT":
      cost_function = pyismtools.utils.chi2_ISMFIT
      
   # prepare the bounds and the initial guess for parameter if not provided
   l_mins_array = np.log10(my_approximator_cp.get_parameter_mins())
   l_maxs_array = np.log10(my_approximator_cp.get_parameter_maxs())
   l_bounds = [(l_mins_array[i],l_maxs_array[i]) for i in range(my_approximator_cp.get_N_parameters())]
   
   assert initial_parameters==None or isinstance(initial_parameters,Model) or (isinstance(initial_parameters,np.ndarray) and initial_parameters.ndim == 1 and initial_parameters.shape[0] == my_approximator.get_N_parameters()), "Argument \"initial_parameters\" must be either a numpy array of dimension 1 and with as many values as there are parameters in the provided Grid_approximator, or a Model object."
   model_init_scaling_factor = None
   if initial_parameters == None :
      initial_parameters = 10** ((l_maxs_array+l_mins_array)/2.) # take the log average of the min and max on the grid
   elif isinstance(initial_parameters,Model): # if a model is provided as initial guess extract the parameter values and scaling factor
      if with_scaling_factor==True:
         model_init_scaling_factor = initial_parameters.get_scaling_factor()
      initial_parameters = initial_parameters.get_parameter_vector()
   
   # If the selected method does not accept bounds, we allow extrapolation in order to enforce softbounds
   allow_extrapolation = (method not in ["Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr", "COBYLA"])
   
   # initial guess plus specific preparation if scaling factor
   if with_scaling_factor==True:
      if model_init_scaling_factor is not None:
         initial_scaling_factor = model_init_scaling_factor
      l_initial_guess = np.array([initial_scaling_factor] + list(np.log10(initial_parameters)))
      l_bounds = [tuple(scaling_factor_bounds)] + l_bounds
      l_bounds = np.array(l_bounds)
      def temp_cost_function(l_param_vector,l_bounds):
         scaling_factor = l_param_vector[0]
         l_true_param_vector = l_param_vector[1:]
         return_value = cost_function(scaling_factor*my_approximator_cp(10**l_true_param_vector,allow_extrapolation=allow_extrapolation).get_observable_vector(),my_observation_cp.get_observable_vector(),my_observation_cp.get_error_vector(),my_observation_cp.get_error_type_vector(),my_approximator_cp.get_N_parameters())
         if allow_extrapolation : # method does not accept bounds, add a bound penalty term to (softly) enforce the bounds
            return_value += pyismtools.utils.out_of_bound_penalty(l_param_vector,l_bounds)
         return return_value
   else :
      l_initial_guess = np.log10(initial_parameters)
      def temp_cost_function(l_param_vector,l_bounds):
         return_value = cost_function(my_approximator_cp(10**l_param_vector,allow_extrapolation=allow_extrapolation).get_observable_vector(),my_observation_cp.get_observable_vector(),my_observation_cp.get_error_vector(),my_observation_cp.get_error_type_vector(),my_approximator_cp.get_N_parameters())
         if allow_extrapolation : # method does not accept bounds, add a bound penalty term to (softly) enforce the bounds
            return_value += pyismtools.utils.out_of_bound_penalty(l_param_vector,l_bounds)
         return return_value

   # Perform the minimization
   res_list = []
   chi2_list = []
   fit_success_list = []

   # repeat minimiation N_repetitions times
   my_range = tqdm(range(N_repetitions)) if not quiet else range(N_repetitions)
   for i in my_range:
      # redefine initial conditions : repeated try from random guesses after the first try which always uses the provided initial guess
      if (i>0):
         if with_scaling_factor==True:
            l_initial_guess = [np.random.uniform(scaling_factor_bounds[0],scaling_factor_bounds[1])] + [np.random.uniform(l_mins_array[i],l_maxs_array[i]) for i in range(my_approximator_cp.get_N_parameters())]
         else :
            l_initial_guess = [np.random.uniform(l_mins_array[i],l_maxs_array[i]) for i in range(my_approximator_cp.get_N_parameters())]
      # Do minimization
      if allow_extrapolation : # method does not accept bounds (but we have included soft bounds in the cost function)
         temp_res = scipy.optimize.minimize(temp_cost_function, l_initial_guess,method=method,args=(l_bounds))
      else : # method accepts bounds
         temp_res = scipy.optimize.minimize(temp_cost_function, l_initial_guess,bounds=l_bounds,method=method,args=(l_bounds))

      best_chi2 = temp_cost_function(temp_res.x,l_bounds)[0]
      if N_repetitions == 1 and not temp_res.success :
         print("Fit unsuccessful : %s\nTry improving your initial guess, or changing the minimization method."%(temp_res.message))
      res_list.append(temp_res.x)
      chi2_list.append(best_chi2)
      fit_success_list.append(temp_res.success)
      if not quiet :
         print("Restart %i"%i,", best chi2 found = %.3f"%best_chi2+("" if temp_res.success else ", failure: "+temp_res.message))

   if N_repetitions>1 and not quiet :
      print("%i out of the %i restarts converged successfully.\n"%(np.count_nonzero(fit_success_list),N_repetitions))
      
   ind_best_mod = chi2_list.index(np.nanmin(chi2_list))
   if with_scaling_factor==True:
      scaling_factor = res_list[ind_best_mod][0]
      param_vector = 10**res_list[ind_best_mod][1:]
      best_model = my_approximator(param_vector,allow_extrapolation=allow_extrapolation) # use the original approximator to have all observables
      best_model.set_scaling_factor(scaling_factor)
   else :
      best_model = my_approximator(10**res_list[ind_best_mod],allow_extrapolation=allow_extrapolation) # use the original approximator to have all observables
   
   return best_model, chi2_list[ind_best_mod]

def find_best_interpolated_model_map(my_observation_map,my_approximator,initial_parameters=None,cost_function="chi2",with_scaling_factor=True,initial_scaling_factor=1.,scaling_factor_bounds=[0.1,10.],method="SLSQP",N_repetitions = 1):
   best_model_list = []
   best_chi2_list = []
   assert initial_parameters == None or ( isinstance(initial_parameters,list) and len(initial_parameters) == my_observation_map.get_N_pixels() )
   for i in tqdm(range(my_observation_map.get_N_pixels())) :
      init_params = initial_parameters[i] if initial_parameters is not None else None
      observation = my_observation_map.get_observation_list()[i]
      best_model, best_chi2 = find_best_interpolated_model_single_obs(observation,my_approximator,initial_parameters=init_params,cost_function=cost_function,with_scaling_factor=with_scaling_factor,initial_scaling_factor=initial_scaling_factor,scaling_factor_bounds=scaling_factor_bounds,method=method,N_repetitions=N_repetitions,quiet=True)
      best_model_list.append(best_model)
      best_chi2_list.append(best_chi2)
   return best_model_list, best_chi2_list
