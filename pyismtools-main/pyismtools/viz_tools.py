import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyismtools.classes import *
from tqdm.auto import tqdm
from math import floor
from matplotlib import colors
from scipy.interpolate import griddata

def plot_intensity_diagram(item_list,ylabel="Observable",title=None,normalization=None,ylog=True,labels=None,cur_axis=None,colors=None):

   ##### check if all objects have at least one observable in common. Expand observations to union set of all observations, then reduce everyone to intesection with all models
   # As a result, we keep only lines :
   # - that are included in all model objects
   # - that are included in at least one observation object
   
   assert all( ( isinstance(item,Observation) or isinstance(item,Model) ) for item in item_list ), "The function plot_intensity_diagram can only plot objects of type Observation or of type Model."
   
   std_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
   
   item_name_lists = [item.get_observable_names() for item in item_list]
   
   if (len(item_list)>1) and not(all([item_name_list == item_name_lists[0] for item_name_list in item_name_lists])) : # if only one object, nothing to compare. if all objects have same observable name sets in same order, nothing to do as well.
      
      # We keep observables that are in at least one Observation object (unless there are no observation objects)
      obs_name_sets = [set(item.get_observable_names()) for item in item_list if isinstance(item,Observation)]
      if len(obs_name_sets)==0 : # There are no observation objects to plot
         union_list = []
      else : # There are at least one Observation object
         if all([obs_name_set == obs_name_sets[0] for obs_name_set in obs_name_sets]) : # all observation objects have the same observable set :
            union_list = list(obs_name_sets[0])
         else : # the obs name sets are not identical between all objects, so we need to bring them to a common set of observables
            # then, take the union of all observable sets and expand Observation objects (if there are at least 2 Observation)
            union_list = list(set.union(*obs_name_sets)) # taking the union of all name sets
      
      # We keep observables that are in all Model objects (unless there are no Model objects)
      model_name_sets = [set(item.get_observable_names()) for item in item_list if isinstance(item,Model)]
      if len(model_name_sets)==0 : # There are no model objects to plot
         intersection_list = []
      else : # There are at least one Model object
         if all([model_name_set == model_name_sets[0] for model_name_set in model_name_sets]): # all model objects have the same observable set :
            intersection_list = list(model_name_sets[0])
         else : # the model name sets are not identical between all objects, so we need to bring them to a common set of observables
            intersection_list = list(set.intersection(*model_name_sets))
      
      # Finaly, take we keep lines that verify both of the above conditions
      if len(union_list)==0 : # there are no observation objects
         final_list = intersection_list
      elif len(intersection_list)==0 : # There are no model objects
         final_list = union_list
      else : # There are both observations and models to plot
         final_list = list(set.intersection(set(union_list),set(intersection_list)))
      # Bring all the objects to this same observable set, in the same order
      final_list.sort() # for reproducibility of the order of lines on the plot, lines are always sorted by alphabetical order
      # Check that there is something left to plot
      if len(final_list)==0 :
         print("The lists of observables for the listed objects have no common quantities ! There is nothing to plot.")
         return 1
      new_item_list =[]
      for item in item_list :
         if isinstance(item,Observation) :
            item_cpy = item.copy()
            item_cpy.expand_reduce_observables(final_list)
            new_item_list.append(item_cpy)
         else :
            item_cpy = item.copy()
            new_item_list.append(item_cpy)
            item_cpy.reduce_observables(final_list)
      item_list = new_item_list

   ##### produce figure
   if (cur_axis is None):  # if no axes provided, create a new figure and get the axes, otherwise we will work on the provided axes
        fig = plt.figure(figsize=(12,9))
        cur_axis = plt.gca()
   else:
        fig = cur_axis.figure

   for i_item, item in enumerate(item_list):
      if isinstance(item,Model):
         model = item.copy()
         if normalization is not None :
            norm_value = model.get_observable_value(normalization)
            temp_name_list = model.get_observable_names()
            temp_name_list.remove(normalization)
            model.reduce_observables(temp_name_list)
         else :
            norm_value = 1.
         param_names = model.get_parameter_names()
         param_vector = model.get_parameter_vector()
         obs_names = model.get_observable_names()
         obs_vector = model.get_observable_vector()
         scaling_factor = model.get_scaling_factor()
         if labels is None :
            label = r"Model :" if not model.interpolated_model else "Interpolated model : "
            for i in range(len(param_names)):
               param_value = param_vector[i]
               param_exp = floor(np.log10(param_vector[i]))
               param_prefact = param_vector[i]/(10.**param_exp)
               if param_prefact!=1. and param_exp!=0 :
                  param_value = r"$%.2g \times 10^{%i}$"%(param_prefact,param_exp)
               elif param_prefact==1 :
                  param_value = r"$10^{%i}$"%(param_exp)
               else : # param_exp == 0
                  param_value = r"$%.2g$"%(param_prefact)
               label += " " + param_names[i] + " = %s,"%param_value
            if scaling_factor != 1. :
               label += " scaling factor = %.2g,"%scaling_factor
            label = label[:-1] # remove the last character (a comma).
         else :
            label = labels[i_item]
         color = colors[i_item] if colors is not None else std_color_cycle[i_item] # use user specified colors if provided else use standard color cycle
         cur_axis.plot(range(len(obs_names)),obs_vector/norm_value,'o-',label=label,color=color)
      elif isinstance(item,Observation):
         obs = item.copy()
         if normalization is not None :
            norm_value = obs.get_observable_value(normalization)
            temp_name_list = obs.get_observable_names()
            temp_name_list.remove(normalization)
            obs.reduce_observables(temp_name_list)
         else :
            norm_value = 1.
         obs_names = obs.get_observable_names()
         obs_vector = obs.get_observable_vector()
         obs_stds = obs.get_error_vector()
         obs_err_type = obs.get_error_type_vector()
         if labels is None :
            label = f"Observation : {obs.observation_id:20}"
         else :
            label = labels[i_item]
         errbars = np.zeros((2,obs.get_N_observables()))
         mask_normal = (obs_err_type=="normal")
         mask_lognormal = (obs_err_type=="lognormal")
         if np.any(mask_normal):
            errbars[:,mask_normal] = np.vstack( ( (obs_stds/norm_value)[mask_normal], (obs_stds/norm_value)[mask_normal] ) )
         if np.any(mask_lognormal):
            errbars[:,mask_lognormal] = np.vstack( ( ((1.-1./obs_stds)*obs_vector/norm_value)[mask_lognormal] , ((obs_stds-1.)*obs_vector/norm_value)[mask_lognormal] ) )
         color = colors[i_item] if colors is not None else std_color_cycle[i_item] # use user specified colors if provided else use standard color cycle
         cur_axis.errorbar(range(len(obs_names)),obs_vector/norm_value,yerr=errbars,label=label,fmt='s',capsize=3,color=color)
      else :
         print("plot_intensity_diagram cannot plot object of type {type(item)}. Skipping this object.")
   if title is not None :
      cur_axis.set_title(title,fontsize=20)
   if normalization is not None :
      ylabel += f"(normalized to {normalization:s})"
   cur_axis.set_ylabel(ylabel,fontsize=20)
   cur_axis.set_xticks(range(len(obs_names)), obs_names, rotation='vertical',fontsize=11)
   cur_axis.set_xlim(-1,len(obs_names))
   if ylog :
      cur_axis.set_yscale('log')
   cur_axis.legend(fontsize=12)
   
   if (cur_axis is None):
      fig.tight_layout()
   
   return cur_axis
   
def plot_intensity_maps(observation_map,title=None,normalization=None,logscale=True,cmap="viridis",cur_figure=None):

   observable_names = observation_map.get_observable_names()
      
   # Make figures
   n_cols = 3
   n_rows = len(observable_names)//n_cols + (1 if len(observable_names)%n_cols !=0 else 0)
   
   if (cur_figure is None):  # if no figure provided, create a new figure, otherwise we will work on the provided figure
      cur_figure = plt.figure(figsize=(15,9*n_rows))
      
   for i,observable_name in enumerate(observable_names) :
      cur_axis = cur_figure.add_subplot(n_rows,n_cols,i+1)
      plot_intensity_maps_singleline(observation_map,observable_name,logscale=logscale,cmap=cmap,cur_axis=cur_axis)
      
   cur_figure.tight_layout()
   
   return cur_figure
   
def plot_intensity_maps_singleline(observation_map,observable_name,logscale=True,cmap="viridis",cur_axis=None):
   # reconstruct map dimensions
   pixel_indices = observation_map.get_pixel_index_list()
   max_i = max(pixel_indices,key=lambda x:x[1])[1]
   max_j = max(pixel_indices,key=lambda x:x[0])[0]
   
   # build 2D map
   values_list = observation_map.get_single_observable_map(observable_name)
   val_map = np.nan*np.ones((max_i+1,max_j+1))
   for i,indices in enumerate(pixel_indices):
      val_map[indices[1],indices[0]] = values_list[i]
      
   if (cur_axis is None):  # if no axes provided, create a new figure and get the axes, otherwise we will work on the provided axes
      fig = plt.figure(figsize=(12,9))
      cur_axis = plt.gca()   
   
   if logscale :
      im = cur_axis.imshow(val_map,origin="lower",norm=colors.LogNorm(),cmap=cmap)
   else :
      im = cur_axis.imshow(val_map,origin="lower",cmap=cmap)
   CS = plt.colorbar(im)
   CS.set_label(observable_name,fontsize=20)
   
   if (cur_axis is None):
      plt.tight_layout()
      
   return cur_axis
   
def fit_quality_contour_plots(grid_approximator,observation,ref_model,with_scaling_factor=True,cost_function="chi2",scaling_factor_bounds
                          =
                          [0.1,10.],n_contours=10,uncertainty_contour=False,use_true_Dchi2_threshold=False,cmap="viridis",chi2_bounds=[None,None],cur_figure=None,with_diag=True):

   # Reduce grid_approximator and ref_model to the set of observables of the observation
   observable_list = observation.get_observable_names()
   grid_approximator = grid_approximator.copy()
   ref_model = ref_model.copy()
   grid_approximator.reduce_observables(observable_list)
   ref_model.reduce_observables(observable_list)

   # Prepare useful quantities
   nb_params = grid_approximator.get_N_parameters() + with_scaling_factor
   
   # set number of plots
   n1, n2 = (nb_params,nb_params) if with_diag else (nb_params-1,nb_params-1)

   # build figure
   if (cur_figure is None):  # if no figure provided, create a new figure, otherwise we will work on the provided figure
      cur_figure, axs =  plt.subplots(n1,n2,figsize=(7*nb_params,6*nb_params))
   else :
      axs = cur_figure.subplots(n1,n2)
   
   # plot the different panels
   for k in tqdm(range(nb_params**2)):
      i = k//nb_params
      j = k%nb_params
      param_name_1 = "Scaling factor" if (i==0 and with_scaling_factor) else grid_approximator.get_parameter_names()[i-with_scaling_factor]
      param_name_2 = "Scaling factor" if (j==0 and with_scaling_factor) else grid_approximator.get_parameter_names()[j-with_scaling_factor]
      parameter_name_couple = (param_name_1,param_name_2)
      
      if j>i : # plots above the diagonal are just empty
         if with_diag :
            axs[i,j].axis('off')
         elif (i>0 and j<nb_params-1) :
            axs[i-1,j].axis('off')
      elif j<i or with_diag :
         if with_diag :
            cur_axis = axs[i,j]
         else :
            cur_axis = axs[i-1,j]
         fit_quality_contour_plots_singlepair(grid_approximator,observation,parameter_name_couple,ref_model,with_scaling_factor=with_scaling_factor,cost_function=cost_function,scaling_factor_bounds=scaling_factor_bounds,n_contours=n_contours,uncertainty_contour=uncertainty_contour,use_true_Dchi2_threshold=use_true_Dchi2_threshold,cmap=cmap,chi2_bounds=chi2_bounds,cur_axis=cur_axis)
      elif i==j and not with_diag and (i>0 and j<nb_params-1) :
         axs[i-1,j].axis('off')
   
   # figure.tight_layout escape from the thread safe lock in Figure. Use
   # set_layout_engine instead, which doesn't trigger a bypass of the Figure
   # RLock.
   # Plus, tight_layout() deprecated since 3.6
   cur_figure.set_layout_engine('tight')

   return cur_figure
   
def fit_quality_contour_plots_singlepair(grid_approximator,observation,parameter_name_couple,ref_model,with_scaling_factor=True,cost_function="chi2",scaling_factor_bounds=[0.1,10.],n_contours=10,uncertainty_contour=False,use_true_Dchi2_threshold=False,cmap="viridis",chi2_bounds=[None,None],cur_axis=None):
   
   # prepare chi2 thresholds
   if use_true_Dchi2_threshold :
      thresholds = [1., 2.3, 3.53, 4.72, 5.89, 7.04] # delta chi2 defining the 68.27% uncertainty domain depending on number of degrees of freedom, in a normal approximation (from Numerical Recipes, 15.6.5) - Probably wrong, check again
   else :
      thresholds = [1., 1., 1., 1., 1., 1.] # just always show the contour with Delta chi2 = 1
   
   # plot parameters
   font_size = 18
   n_points = 50
   
   # translate string for cost function if given
   if cost_function == "chi2":
      my_cost_function = pyismtools.utils.chi2
   elif cost_function == "reduced_chi2":
      my_cost_function = pyismtools.utils.reduced_chi2
   elif cost_function == "chi2_ISMFIT":
      my_cost_function = pyismtools.utils.chi2_ISMFIT
      
   # Prepare mask on undetected obs values
   nan_mask = np.isnan(observation.get_observable_vector())
   
   # best model
   best_chi2 = my_cost_function(ref_model.get_observable_vector()[~nan_mask],observation.get_observable_vector()[~nan_mask],observation.get_error_vector()[~nan_mask],observation.get_error_type_vector()[~nan_mask],ref_model.get_N_parameters())
   ref_parameters = np.array( ([ref_model.get_scaling_factor()] if with_scaling_factor else []) + list(ref_model.get_parameter_vector()) )
   
   # prepare grids of parameter values for the plots
   nb_params = grid_approximator.get_N_parameters() + with_scaling_factor
   param_grids = []
   for param_name in parameter_name_couple :
      if param_name == "Scaling factor" :
         param_grids.append(np.logspace(np.log10(scaling_factor_bounds[0]),np.log10(scaling_factor_bounds[1]),n_points))
      else :
         i_param = grid_approximator.get_parameter_names().index(param_name)
         param_grids.append(np.logspace(np.log10(grid_approximator.get_parameter_mins()[i_param]),np.log10(grid_approximator.get_parameter_maxs()[i_param]),n_points))
   
   param_name_1, param_name_2 = parameter_name_couple
   
   if (cur_axis is None):  # if no axes provided, create a new figure and get the axes, otherwise we will work on the provided axes
      fig = plt.figure(figsize=(12,9))
      cur_axis = plt.gca()  
   
   # if the two parameters are the same :  1D cuts of the cost function vs one parameter
   if param_name_1==param_name_2 :
      i_param = grid_approximator.get_parameter_names().index(param_name_1)+1 if param_name_1 != "Scaling factor" else 0
      temp_param_grid = np.tile(ref_parameters,(n_points,1))
      temp_param_grid[:,i_param] = param_grids[0]
      cost_grid = my_cost_function(temp_param_grid[:,0][:,np.newaxis]*grid_approximator.vectorized_call(temp_param_grid[:,1:])[:,~nan_mask],observation.get_observable_vector()[~nan_mask],observation.get_error_vector()[~nan_mask],observation.get_error_type_vector()[~nan_mask],grid_approximator.get_N_parameters())
      divider = make_axes_locatable(cur_axis)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      cax.axis('off')
      cur_axis.plot(param_grids[0],cost_grid)
      cur_axis.set_xlabel(param_name_1,fontsize=font_size)
      cur_axis.set_ylabel(cost_function,fontsize=font_size)
      cur_axis.set_xscale('log')
      cur_axis.set_yscale('log')
      cur_axis.axvline(ref_parameters[i_param],linestyle="--",color='r')
   
   # otherwise : 2D contour plots for the cost function vs two parameters
   else :
      # compute the cost function on the 2D grid
      i_param1 = grid_approximator.get_parameter_names().index(param_name_1)+1 if param_name_1 != "Scaling factor" else 0
      i_param2 = grid_approximator.get_parameter_names().index(param_name_2)+1 if param_name_2 != "Scaling factor" else 0
      temp_param_grid = np.tile(ref_parameters,(n_points,n_points,1))
      temp_param_grid[:,:,i_param1] = param_grids[0][:,np.newaxis]
      temp_param_grid[:,:,i_param2] = param_grids[1][np.newaxis,:]
      flattened_param_grid = np.reshape(temp_param_grid,(-1,temp_param_grid.shape[-1])) # turn a (k,l,m) array into a (k*l,m) array (flattening the first two axis only, so that we have a list of parameter vectors)
      flattened_cost_map = my_cost_function(flattened_param_grid[:,0][:,np.newaxis]*grid_approximator.vectorized_call(flattened_param_grid[:,1:])[:,~nan_mask],observation.get_observable_vector()[~nan_mask],observation.get_error_vector()[~nan_mask],observation.get_error_type_vector()[~nan_mask],grid_approximator.get_N_parameters())
      cost_map = np.reshape(flattened_cost_map,(n_points,n_points)) # reshape into a map
      # make the plot
      divider = make_axes_locatable(cur_axis)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      cur_axis.set_xscale('log')
      cur_axis.set_yscale('log')
      vmin = np.nanmin(cost_map[np.nonzero(cost_map)])
      vmax = np.nanmax(cost_map[np.nonzero(cost_map)])
      if(chi2_bounds[0] is not None) :
         vmin = chi2_bounds[0]
      if(chi2_bounds[1] is not None) :
         vmax = chi2_bounds[1]
      if ( floor(np.log10(vmax)) - floor(np.log10(vmin)) - 1 < 1 ):
         vmin = 10.**floor(np.log10(vmin))
         vmax = 10.**floor(np.log10(vmax)+1)
      elif ( floor(np.log10(vmax)) - floor(np.log10(vmin)) - 1 < 2 ):
         vmin = 10.**floor(np.log10(vmin))
      cs = cur_axis.contourf(param_grids[1],param_grids[0],cost_map,norm = LogNorm(),locator=ticker.LogLocator(),vmin=vmin,vmax=vmax,cmap=cmap,levels=np.logspace(np.log10(vmin),np.log10(vmax),200),zorder=0)
      cur_axis.contour(param_grids[1],param_grids[0],cost_map,norm = LogNorm(),locator=ticker.LogLocator(),vmin=vmin,vmax=vmax,cmap=cmap,levels=np.logspace(np.log10(vmin),np.log10(vmax),200),zorder=0)
      cs2 = cur_axis.contour(param_grids[1],param_grids[0],cost_map,norm = LogNorm(),locator=ticker.LogLocator(),vmin=vmin,vmax=vmax,colors="w",levels=np.logspace(np.log10(vmin),np.log10(vmax),n_contours),zorder=1,linestyles=":")
      labels = cur_axis.clabel(cs2, cs2.levels, fontsize=13, inline = True,
                              inline_spacing = 2) 

      for l in labels:
         l.set_rotation(0)
      cur_axis.scatter(ref_parameters[i_param2],ref_parameters[i_param1],c='tab:orange',s=100,zorder=3)
      fig = cur_axis.get_figure()
      cbar = fig.colorbar(cs, cax = cax, orientation='vertical')
      cbar.ax.set_yscale('log')
      cur_axis.set_ylabel(param_name_1,fontsize=font_size)
      cur_axis.set_xlabel(param_name_2,fontsize=font_size)
      if nb_params<7 and cost_function == "reduced_chi2" and uncertainty_contour :
            cur_axis.contour(param_grids[1],param_grids[0],y,levels=[best_chi2 + thresholds[nb_params-1]],colors=['orange'],zorder=2)
   
   if (cur_axis is None):
      plt.tight_layout()
   
   return cur_axis
   
def map_fit_quality(observation_map,chi2_list,colorbar_label=r"$\chi^2$",logscale=True,cmap="viridis",cur_axis=None):
   pixel_index_list = observation_map.get_pixel_index_list()
   # prepare chi2 map
   # max_i = max(pixel_index_list,key=lambda x:x[0])[0]
   # max_j = max(pixel_index_list,key=lambda x:x[1])[1]
   max_i = max(x for y, x in pixel_index_list)
   max_j = max(y for y, x in pixel_index_list)

   chi2_map = np.nan*np.ones((max_i+1,max_j+1))
   for i,indices in enumerate(pixel_index_list):
      chi2_map[indices[1],indices[0]] = chi2_list[i]
      
   # produce figure
   if (cur_axis is None):  # if no axes provided, create a new figure and get the axes, otherwise we will work on the provided axes
      fig = plt.figure(figsize=(12,9))
      cur_axis = plt.gca()
    
   if logscale:
      im = cur_axis.imshow(chi2_map,origin="lower",norm=colors.LogNorm(),cmap=cmap)
   else :
      im = cur_axis.imshow(chi2_map,origin="lower",cmap=cmap)

   from mpl_toolkits.axes_grid1 import make_axes_locatable
   divider = make_axes_locatable(cur_axis)
   cax = divider.append_axes("right", size = "5%", pad = 0.05)
   cs = cur_axis.figure.colorbar(im, cax = cax, aspect = 10 ) 
   cs.set_label(colorbar_label, fontsize=20)
   
   if (cur_axis is None):
      fig.tight_layout()
   
   return cur_axis

def map_best_parameters(observation_map,best_model_list,logscale=True,cmap="viridis",cur_figure=None):
   # get list of parameters
   parameter_list = best_model_list[0].get_parameter_names()
   assert all(model.get_parameter_names() == parameter_list for model in best_model_list), "Second argument to function map_best_parameters must be a list of models that all have the same list of parameter names."

   # produce figure
   n_cols = 3
   n_rows = len(parameter_list)//n_cols + (1 if len(parameter_list)%n_cols !=0 else 0)
   
   if (cur_figure is None):  # if no figure provided, create a new figure, otherwise we will work on the provided figure
      cur_figure = plt.figure(figsize=(15,9*n_rows))

   # loop on parameters
   for i,parameter_name in enumerate(parameter_list) :
      cur_axis = cur_figure.add_subplot(n_rows,n_cols,i+1)
      map_best_parameter_singleparam(observation_map,best_model_list,parameter_name,logscale=logscale,cmap=cmap,cur_axis=cur_axis)
      
   cur_figure.tight_layout()
   
   return cur_figure
   
def map_best_parameter_singleparam(observation_map,best_model_list,parameter_name,logscale=True,cmap="viridis",cur_axis=None):
   # reconstruct map dimensions
   pixel_index_list = observation_map.get_pixel_index_list()
   max_i = max(pixel_index_list,key=lambda x:x[1])[1]
   max_j = max(pixel_index_list,key=lambda x:x[0])[0]
   
   # produce figure
   if (cur_axis is None):  # if no axes provided, create a new figure and get the axes, otherwise we will work on the provided axes
      fig = plt.figure(figsize=(12,9))
      cur_axis = plt.gca()
   
   # build 2D map of the parameter
   param_map = np.nan*np.ones((max_i+1,max_j+1))
   for i,indices in enumerate(pixel_index_list):
      param_map[indices[1], indices[0]] = best_model_list[i].get_parameter_value(parameter_name)
   if logscale:
      im = cur_axis.imshow(param_map,origin="lower",norm=colors.LogNorm(),cmap=cmap)
   else:
      im = cur_axis.imshow(param_map,origin="lower",cmap=cmap)

   from mpl_toolkits.axes_grid1 import make_axes_locatable
   divider = make_axes_locatable(cur_axis)
   cax = divider.append_axes("right", size = "5%", pad = 0.05)
   cur_axis.figure.colorbar(im, cax = cax, aspect = 10 ) 

   if (cur_axis is None):
      fig.tight_layout()
      
   return cur_axis

def observable_contour_plot(model_grid,observable_name,fixed_parameter_names_and_values,scaling_factor=1.,cmap="viridis",cur_axis=None):

   fixed_parameter_names = list(fixed_parameter_names_and_values.keys())
   # check that all parameters to fix are indeed in the grid
   for param_name in fixed_parameter_names :
      assert (param_name in model_grid.get_parameter_names()), "Cannot fix parameter %s as it is not present in the grid."%(param_name)
   assert model_grid.get_N_parameters() - len(fixed_parameter_names) ==  2, "After fixing the requested parameters, there should remain exactly 2 free parameters (%i remaining)"%(model_grid.get_N_parameters - len(fixed_parameter_names))
      
      
   temp_grid = model_grid.copy()
   temp_grid.reduce_observables([observable_name])
   temp_grid.reduce_parameters(fixed_parameter_names_and_values)
   free_parameter_names = temp_grid.get_parameter_names()
   param_array = temp_grid.get_parameter_array()
   obs_array = temp_grid.get_observable_array()
   
   # interpolate values on finer grid
   log_xgrid = np.linspace(np.log10(param_array[:,0].min()),np.log10(param_array[:,0].max()),100)
   log_ygrid = np.linspace(np.log10(param_array[:,1].min()),np.log10(param_array[:,1].max()),100)
   log_z_interp = griddata((np.log10(param_array[:,0]), np.log10(param_array[:,1])), np.log10(scaling_factor*obs_array[:,0]), (log_xgrid[None, :], log_ygrid[:, None]), method='cubic')
   
   # Make contour plot
   if (cur_axis is None):  # if no axes provided, create a new figure and get the axes, otherwise we will work on the provided axes
      fig = plt.figure(figsize=(12,9))
      cur_axis = fig.add_subplot(1,1,1)
   cur_axis.set_xscale("log")
   cur_axis.set_yscale("log")
   CS1 = cur_axis.contourf(10**log_xgrid,10**log_ygrid,10**log_z_interp,np.logspace(np.nanmin(log_z_interp),np.nanmax(log_z_interp),100),locator=plt.LogLocator(),cmap=cmap)
   cb = cur_axis.get_figure().colorbar(CS1)
   cb.ax.set_yscale('log') # fixes current matplotlib bug - might not be necessary with future version of matplotlib
   line_contours = np.logspace(np.floor(np.nanmin(log_z_interp)),np.floor(np.nanmax(log_z_interp))+1,int(np.floor(np.nanmax(log_z_interp))+2-np.floor(np.nanmin(log_z_interp))))
   CS2 = cur_axis.contour(10**log_xgrid,10**log_ygrid,10**log_z_interp,line_contours,locator=ticker.LogLocator(),colors="w")
   
   fmt = ticker.LogFormatterMathtext()
   fmt.create_dummy_axis()
   labels = cur_axis.clabel(CS2, CS2.levels, fontsize=10, fmt=fmt , inline = True, inline_spacing = 10)

   for l in labels:
     l.set_rotation(0)
   
   cur_axis.scatter(param_array[:,0],param_array[:,1],c='lightgrey',alpha=0.7)
   cb.set_label(observable_name, fontsize=20)
   cur_axis.set_xlabel(free_parameter_names[0],fontsize=20)
   cur_axis.set_ylabel(free_parameter_names[1],fontsize=20)
   
   return cur_axis
   
def add_bounds_to_observable_contour_plot(model_grid,fixed_parameter_names_and_values,bounds_dict,cur_axis,scaling_factor=1.,color_list=None):
   assert (type(bounds_dict) is dict), '"bounds_dict" argument should receive a dict associating a tuple of bounds to an observable name.'
   
   # Select the color that will be used to display the bound
   if color_list is None :
      color_list = plt.rcParams['axes.prop_cycle'].by_key()['color'] # should be changed to a list of more adequate colors
   index_color = 0
   
   temp_grid = model_grid.copy()
   temp_grid.reduce_parameters(fixed_parameter_names_and_values)
   temp_grid.reduce_observables(bounds_dict.keys())
   param_array = temp_grid.get_parameter_array()
   obs_array = temp_grid.get_observable_array()
   
   # finer grids for interpolations
   log_xgrid = np.linspace(np.log10(param_array[:,0].min()),np.log10(param_array[:,0].max()),100)
   log_ygrid = np.linspace(np.log10(param_array[:,1].min()),np.log10(param_array[:,1].max()),100)
   
   for observable_name in bounds_dict.keys() :
      new_color = color_list[index_color]
      obs_index = temp_grid.observable_names.index(observable_name)
      bounds = list(bounds_dict[observable_name])
      if bounds[0] is None :
         bounds[0] = - np.inf
      if bounds[1] is None :
         bounds[1] = np.inf
      # interpolate predicted values on finer grid
      log_z_interp = griddata((np.log10(param_array[:,0]), np.log10(param_array[:,1])), np.log10(scaling_factor*obs_array[:,obs_index]), (log_xgrid[None, :], log_ygrid[:, None]), method='cubic')
      cur_axis.contour(10**log_xgrid, 10**log_ygrid, 10**log_z_interp, levels=[bounds[0], bounds[1]],colors=new_color,linewidth=3)
      cur_axis.contourf(10**log_xgrid, 10**log_ygrid, 10**log_z_interp, levels=[bounds[0], bounds[1]],cmap=ListedColormap([new_color]),alpha=0.5)
      index_color = (index_color + 1)%len(color_list)
      
   return cur_axis
   
def observable_scatter_plot(model_grid,observable_names,color_param=None,cur_axis=None):
   assert len(observable_names)==2, "Can only plot scatter of two observable quantities (%i given)."%(len(observable_names))
   
   quants = model_grid.get_observable_array(observable_list = observable_names)
   if color_param is not None :
      color_quant = model_grid.get_parameter_array(parameter_list=[color_param])
   
   if (cur_axis is None):  # if no axes provided, create a new figure and get the axes, otherwise we will work on the provided axes
      fig = plt.figure(figsize=(12,9))
      cur_axis = plt.gca()
      
   if color_param is None :
      sca = cur_axis.scatter(quants[:,0],quants[:,1])
   else :
      sca = cur_axis.scatter(quants[:,0],quants[:,1],c=color_quant,norm=LogNorm())
      cb = plt.colorbar(sca)
      cb.set_label(color_param,fontsize=20)
   cur_axis.set_xlabel(observable_names[0],fontsize=20)
   cur_axis.set_ylabel(observable_names[1],fontsize=20)
   cur_axis.set_xscale('log')
   cur_axis.set_yscale('log')
   
   return cur_axis
   
