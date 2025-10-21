import numpy as np
import scipy.interpolate
import itertools

# distance functions

def distance_euclidian(point1,point2):
   return np.linalg.norm(point1 - point2,axis=1)
   
def distance_log_euclidian(point1,point2):
   return np.linalg.norm(np.log10(point1) - np.log10(point2),axis=1)

# cost function for fits

def chi2(grid_observable_array,observed_values,observed_stds,observed_error_pdf,n_params):
   observed_error_pdf = np.array(observed_error_pdf)
   # Always bring model obs to a 2D array (to handle the case where we want to compute the cost function on a grid of models at once)
   if (grid_observable_array.ndim ==1 ):
      grid_observable_array = grid_observable_array[np.newaxis,:]
   # Limit comparison to observables that are not NaN in the observation
   nan_mask = np.isnan(observed_values)
   grid_observable_array = grid_observable_array[:,~nan_mask]
   observed_values = observed_values[~nan_mask]
   observed_stds = observed_stds[~nan_mask]
   observed_error_pdf = observed_error_pdf[~nan_mask]
   # For observables with lognormal error, we do the comparison on the log of the value (which thus has normal errors)
   observed_values[observed_error_pdf == "lognormal"] = np.log10(observed_values)[observed_error_pdf == "lognormal"]
   observed_stds[observed_error_pdf == "lognormal"] = np.log10(observed_stds)[observed_error_pdf == "lognormal"]
   grid_observable_array[:,observed_error_pdf == "lognormal"] = np.log10(grid_observable_array[:,observed_error_pdf == "lognormal"])
   # Finaly, compute the chi2
   return np.sum( (grid_observable_array - observed_values[np.newaxis,:])**2 / observed_stds[np.newaxis,:]**2 , axis=1)

def reduced_chi2(grid_observable_array,observed_values,observed_stds,observed_error_pdf,n_params):
   assert (len(observed_values) > n_params), "Reduced chi2 cost function is only possible with more observables than free parameters."
   observed_error_pdf = np.array(observed_error_pdf) # TODO : ensure this is stored as a np array directly in the class ?
   # Always bring model obs to a 2D array (to handle the case where we want to compute the cost function on a grid of models at once)
   if (grid_observable_array.ndim ==1 ):
      grid_observable_array = grid_observable_array[np.newaxis,:]
   # Limit comparison to observables that are not NaN in the observation
   nan_mask = np.isnan(observed_values)
   grid_observable_array = grid_observable_array[:,~nan_mask]
   observed_values = observed_values[~nan_mask]
   observed_stds = observed_stds[~nan_mask]
   observed_error_pdf = observed_error_pdf[~nan_mask]
   # For observables with lognormal error, we do the comparison on the log of the value (which thus has normal errors)
   observed_values[observed_error_pdf == "lognormal"] = np.log10(observed_values)[observed_error_pdf == "lognormal"]
   grid_observable_array[:,observed_error_pdf == "lognormal"] = np.log10(grid_observable_array[:,observed_error_pdf == "lognormal"])
   observed_stds[observed_error_pdf == "lognormal"] = np.log10(observed_stds)[observed_error_pdf == "lognormal"]
   # Finaly, compute the chi2
   return np.sum( ((grid_observable_array - observed_values[np.newaxis,:])**2 / observed_stds[np.newaxis,:]**2) , axis=1) / (len(observed_values) - n_params)
   
def chi2_ISMFIT(grid_observable_array,observed_values,observed_stds,observed_error_pdf,n_params):
   if np.any(np.array(observed_error_pdf)=="lognormal"):
      raise RuntimeError("ISMFIT cost function does not work with lognormal error distributions.")
   # Always bring model obs to a 2D array (to handle the case where we want to compute the cost function on a grid of models at once)
   if grid_observable_array.ndim == 1 :
      grid_observable_array = grid_observable_array[np.newaxis,:]
   # Limit comparison to observables that are not NaN in the observation
   nan_mask = np.isnan(observed_values)
   grid_observable_array = grid_observable_array[:,~nan_mask]
   observed_values = observed_values[~nan_mask]
   observed_stds = observed_stds[~nan_mask]
   observed_error_pdf = observed_error_pdf[~nan_mask]
   # Compute number of observables
   n_obs = grid_observable_array.shape[1]
   chi2 = np.zeros_like(grid_observable_array)
   # Reshape observation and error vectors to array of same shape as grid_observable_array (duplicate observations for each model in the grid).
   temp_obs = observed_values[np.newaxis,:]*np.ones_like(grid_observable_array) if grid_observable_array.ndim ==2 else observed_values[np.newaxis,:]
   temp_err = observed_stds[np.newaxis,:]*np.ones_like(grid_observable_array) if grid_observable_array.ndim ==2 else observed_stds[np.newaxis,:]
   # Find where model intensity is larger than observed intensity
   mask = grid_observable_array < temp_obs
   # Apply usual chi2 when model intensity < observed intensity
   chi2[mask]  = (grid_observable_array[mask] - temp_obs[mask])**2 / (temp_err[mask]**2)
   # Apply weird modified chi2 when model intensity >= observed intensity
   chi2[~mask] = (( grid_observable_array[~mask] - ( (grid_observable_array[~mask]**2) / temp_obs[~mask]) )**2) /(temp_err[~mask]**2)
   
   # compute the normalisation
   div = n_obs-n_params-1.
   if div <= 0:
      div = 1.
   return np.sum(chi2,axis=1)/div
   
def out_of_bound_penalty(l_param_vector,l_bounds):
   return np.sum((np.maximum(np.maximum(0., l_param_vector - l_bounds[:,1]),l_bounds[:,0]-l_param_vector))**4)

def build_ISMFIT_interpolators(grid_parameters,grid_observables):
   nb_params = grid_parameters.shape[1]
   nb_observables = grid_observables.shape[1]

   # Build Param_val_list (code copied from ISMFIT, changed Model_tab into grid_parameters)
   Param_val_list = []
   for i in range(nb_params):
      # sort out the grid nodes for each flexible parameter
      # [use argsort() and roll the sorted Model_tab to select new nodes.]
      tmparr = grid_parameters[grid_parameters[:, i].argsort(), i]
      #print("tmparr: {}".format(tmparr))
      tmp = ((tmparr/np.roll(tmparr, 1)) != 1.)
      Param_val_list.append(tmparr[tmp])
      
   # Compute the mean log-gridstep (code copied from ISMFIT, changed log to log10)
   param_grid_steps = np.array([np.mean(np.log10(x[1:]) - np.log10(x[:-1])) for x in Param_val_list])
   
   # Build RBF interpolation of the initial grid
   rbfs = []
   for  i in range(nb_observables) :
      # Build a scaled log grid
      tmp = np.log10(grid_parameters)
      for j in range(nb_params):
         tmp[:,j] = tmp[:,j]/param_grid_steps[j]
      
      # Build RBF interpolator on scaled grid
      rbfs.append( scipy.interpolate.RBFInterpolator(tmp,np.log10(grid_observables[:,i]),kernel="quintic") )
      
   
   # Build a regular grid (code copied from ISMFIT, changed log to log10 and Model_tab to grid_parameters)
   
   # Build a 1D regular scaled grid for each parameter
   p_dim = [list(np.unique(np.log10(grid_parameters[:,i])/param_grid_steps[i])) for i in range(nb_params)]

   # interpolate its value using the RBF interpolators and fit a RegularGridInterpolator
   linear_interpolators = []
   for i in range(nb_observables) :
      intensities_grid = np.zeros(tuple([len(grid) for grid in p_dim]))
      for indices in itertools.product(*[range(len(l)) for l in p_dim]):
               intensities_grid[indices] = rbfs[i](np.array([p_dim[k][indices[k]] for k in range(nb_params)])[np.newaxis,:])
      linear_interpolators.append(scipy.interpolate.RegularGridInterpolator(p_dim, intensities_grid, bounds_error=False, fill_value=1e-10))
   
   return linear_interpolators
   
def call_ISMFIT_interpolators(parameter_vector,grid_parameters,linear_interpolators):
   # very innefficient, implemented for comparison with ISMFIT only

   nb_params = grid_parameters.shape[1]
   
   # Build Param_val_list (code copied from ISMFIT, changed Model_tab into grid_parameters)
   Param_val_list = []
   for i in range(nb_params):
      # sort out the grid nodes for each flexible parameter
      # [use argsort() and roll the sorted Model_tab to select new nodes.]
      tmparr = grid_parameters[grid_parameters[:, i].argsort(), i]
      #print("tmparr: {}".format(tmparr))
      tmp = ((tmparr/np.roll(tmparr, 1)) != 1.)
      Param_val_list.append(tmparr[tmp])
      
   # Compute the mean log-gridstep (code copied from ISMFIT, changed log to log10)
   param_grid_steps = np.array([np.mean(np.log10(x[1:]) - np.log10(x[:-1])) for x in Param_val_list])
   
   result_tab = np.array([10**linear_interpolator(np.log10(parameter_vector)/param_grid_steps)[0] for linear_interpolator in linear_interpolators])
   
   return result_tab
   
