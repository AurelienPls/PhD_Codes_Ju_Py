from pyismtools.classes import *
import pymc as pm
import pytensor.tensor as at
from pytensor.compile.ops import as_op
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

def sample_posterior(my_observation,my_approximator,best_model,scaling_factor_bounds=[0.1,10.],nsamples=10000):
   
   param_maxs = my_approximator.get_parameter_maxs()
   param_mins = my_approximator.get_parameter_mins()
   param_names = my_approximator.get_parameter_names()
   init_scaling_factor = best_model.get_scaling_factor()
   init_param_vector = best_model.get_parameter_vector()
   # Shift the initial vector a bit from the edges of the cube as the prior is 0 at the edge -> log prior = nan, which makes pyMC crash if the starting point is on the edge of the cube
   init_param_vector = np.maximum(param_mins*1.001, init_param_vector)
   init_param_vector = np.minimum(param_maxs*0.999, init_param_vector)
   
   observable_names = np.array(my_observation.get_observable_names())
   err_types = np.array(my_observation.get_error_type_vector())
   mask = err_types=="lognormal"
   if (np.any(mask)) and (not np.all(mask)) :
      logn_observable_names = observable_names[err_types=="lognormal"]
      observable_names = observable_names[err_types=="normal"]
      coords = {"param_names": param_names,"obs_names":observable_names,"logn_obs_names":logn_observable_names}
   elif np.all(mask):
      logn_observable_names = observable_names[err_types=="lognormal"]
      coords = {"param_names": param_names,"logn_obs_names":logn_observable_names}
   elif not np.any(mask):
      observable_names = observable_names[err_types=="normal"]
      coords = {"param_names": param_names,"obs_names":observable_names}
   
   # Build bayesian model
   model = pm.Model(coords=coords)
   with model :
      logparameters = pm.Uniform("LogParameters",dims="param_names",lower=np.log10(param_mins),upper=np.log10(param_maxs),initval=np.log10(init_param_vector)) # we use uniform priors on the grid domain
      if init_scaling_factor != 1. :
         scaling_factor = pm.Uniform("scaling_factor",lower=scaling_factor_bounds[0],upper=scaling_factor_bounds[1],initval=init_scaling_factor) # we use a uniform prior on the scaling factor
         # define the model prediction function
         @as_op(itypes=[at.dscalar,at.dvector], otypes=[at.dvector])
         def model(scaling_factor,logparameters):
            return scaling_factor * (my_approximator(10.**logparameters).get_observable_vector())
         # Build observable quantities
            # quantities with normal uncertainties
         if not np.all(mask):
            model_values1 = model(scaling_factor,logparameters)[err_types=="normal"]
            sigma_values1 = my_observation.get_error_vector()[err_types=="normal"]
            obs_values1 = my_observation.get_observable_vector()[err_types=="normal"]
            observables = pm.Normal("Observables",mu=model_values1,sigma=sigma_values1,dims="obs_names",observed=obs_values1)
            # quantities with lognormal uncertainties
         if np.any(mask):
            model_values2 = np.log10(model(scaling_factor,logparameters))[err_types=="lognormal"]
            sigma_values2 = np.log10(my_observation.get_error_vector())[err_types=="lognormal"]
            obs_values2 = np.log10(my_observation.get_observable_vector())[err_types=="lognormal"]
            logobservables = pm.Normal("LogObservables",mu=model_values2,sigma=sigma_values2,dims="logn_obs_names",observed=obs_values2)
      else :
         @as_op(itypes=[at.dscalar,at.dvector], otypes=[at.dvector])
         def model(logparameters):
            return (my_approximator(10.**logparameters).get_observable_vector())
         # Build observable quantities
            # quantities with normal uncertainties
         if not np.all(mask):
            model_values1 = model(logparameters)[err_types=="normal"]
            sigma_values1 = my_observation.get_error_vector()[err_types=="normal"]
            obs_values1 = my_observation.get_observable_vector()[err_types=="normal"]
            observables = pm.Normal("Observables",mu=model_values1,sigma=sigma_values1,dims="obs_names",observed=obs_values1)
            # quantities with lognormal uncertainties
         if np.any(mask):
            model_values2 = np.log10(model(logparameters))[err_types=="lognormal"]
            sigma_values2 = np.log10(my_observation.get_error_vector())[err_types=="lognormal"]
            obs_values2 = np.log10(my_observation.get_observable_vector())[err_types=="lognormal"]
            logobservables = pm.Normal("LogObservables",mu=model_values2,sigma=sigma_values2,dims="logn_obs_names",observed=obs_values2)
      # sample the bayesian model
      idata = pm.sample(draws=nsamples,chains=4,step=pm.Slice([scaling_factor,logparameters]))
      pp = pm.sample_posterior_predictive(idata)
      return idata, pp
      
def diagnostic_figure(sampling_result):
   # Use the ad hoc arviz function
   az.plot_trace(sampling_result, combined=False,compact=False,legend=True)
   # but clean up the figure
   fig = plt.gcf()
   allaxes = fig.get_axes()
   for ax in allaxes :
      label = ax.get_title()
      if '\n' in label :
         label = label.split('\n')[1]
         ax.set_xlabel("log10("+label+")")
      else :
         ax.set_xlabel(label)
      ax.set_title("")
   plt.tight_layout()
   plt.show()
   
def plot_posterior(sampling_result):
   # Use the ad hoc arviz function
   az.plot_posterior(sampling_result)
   # but clean up the figure
   fig = plt.gcf()
   allaxes = fig.get_axes()
   for ax in allaxes :
      label = ax.get_title()
      if '\n' in label :
         label = label.split('\n')[1]
         ax.set_xlabel("log10("+label+")")
      else :
         ax.set_xlabel(label)
      ax.set_title("")
   plt.tight_layout()
   plt.show()
   
def plot_pair_posterior(sampling_result):
   # Use the ad hoc arviz function
   az.plot_pair(sampling_result,marginals=True,kind="kde",point_estimate="mean")
   # but clean up the figure
   # but clean up the figure
   fig = plt.gcf()
   allaxes = fig.get_axes()
   for ax in allaxes :
      label = ax.get_xlabel()
      if '\n' in label :
         label = label.split('\n')[1]
         ax.set_xlabel("log10("+label+")")
      else :
         ax.set_xlabel(label)
      label = ax.get_ylabel()
      if '\n' in label :
         label = label.split('\n')[1]
         ax.set_ylabel("log10("+label+")")
      else :
         ax.set_ylabel(label)
   plt.tight_layout()
   plt.show()
   
def add_posterior_predictions(cur_axis,posterior_predictions,my_observation):
   # get observable name list from Obseravtion object
   original_obs_names = my_observation.get_observable_names()
   obs_err_types = my_observation.get_error_type_vector()
   # generate posterior predictions :
   stacked_posterior_samples = az.extract(posterior_predictions.posterior_predictive)
   # get samples for observables with normal uncertainties
   if np.any(obs_err_types=="normal"):
      sample_data1 = stacked_posterior_samples.Observables.values
      obs_names1 = stacked_posterior_samples.obs_names.values
   # get samples for observables with lognormal uncertainties
   if np.any(obs_err_types=="lognormal"):
      sample_data2 = 10**stacked_posterior_samples.LogObservables.values
      obs_names2 = stacked_posterior_samples.logn_obs_names.values
   # group all samples in a single array with the correct order :
   if np.any(obs_err_types=="normal") and np.any(obs_err_types=="lognormal") :
      sample_data = np.concatenate((sample_data1,sample_data2),axis=0)
      obs_names = np.concatenate((obs_names1,obs_names2),axis=0)
   elif np.all(obs_err_types=="normal"):
      sample_data = sample_data1
      obs_names = obs_names1
   elif np.all(obs_err_types=="lognormal"):
      sample_data = sample_data2
      obs_names = obs_names2
   else :
      err("This case should not be reached.")
   # reorder everything according to the Observation object
   reorder_index = [np.where(obs_names == name)[0][0] for name in original_obs_names]
   obs_names = obs_names[reorder_index]
   sample_data = sample_data[reorder_index,:]
   cur_axis.plot(range(len(obs_names)),sample_data,'-',color=u'#ff7f0e',alpha=0.05,linewidth=0.1)
   
def propagate_posterior(sampling_result,my_func):
   stacked_posterior_samples = az.extract(sampling_result.posterior)
   sample_param_data = stacked_posterior_samples.LogParameters.values
   nbsamples = sample_param_data.shape[1]
   if "scaling_factor" in list(stacked_posterior_samples.variables) :
      sample_scaling_factor_data = stacked_posterior_samples.scaling_factor.values
      sample_data = np.concatenate((sample_scaling_factor_data[np.newaxis,:],sample_param_data),axis=0)
   propagated_sample = [my_func(sample_data[:,i]) for i in range(nbsamples)]
   return np.array(propagated_sample)
      
def get_parameter_covariance_matrix(sampling_result):
   stacked_posterior_samples = az.extract(sampling_result.posterior)
   sample_param_data = stacked_posterior_samples.LogParameters.values
   if "scaling_factor" in list(stacked_posterior_samples.variables) :
      sample_scaling_factor_data = stacked_posterior_samples.scaling_factor.values
      sample_data = np.concatenate((sample_scaling_factor_data[np.newaxis,:],sample_param_data),axis=0)
   cov_mat = np.cov(sample_data)
   return cov_mat
