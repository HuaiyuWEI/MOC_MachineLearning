# Note:
#
# - Search for "def train_model" to locate code related to the Neural Network architecture.
#
# - Other keywords of interest include:
#   - "apply_low_pass_filter" for the Butterworth filter configuration.
#   - "Train loop" for the cross-validation process.
#   - "explainable AI" for implementing layer-wise relevance propagation.
#   - "Use PCA" for employing Empirical Orthogonal Functions (EOFs).
# - This is my first Python code, so please bear with me for the messy code...

import os
import gc
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from keras import regularizers
from keras.models import Model
from keras.layers import Dense,LeakyReLU, Dropout, Add, Activation
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import Callback
from scipy.signal import butter, sosfilt
from matplotlib.colors import Normalize
from sklearn.preprocessing import FunctionTransformer
tf.keras.utils.set_random_seed(0)
tf.config.list_physical_devices('GPU')  
import shap
import innvestigate
tf.compat.v1.disable_eager_execution()
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.engine.training_v1")


#%% Default parameters

# use data from ACCESS_ESM1-5_PIc or GFDL_CM4_PIc
CMIP_name = 'ACCESS_ESM1-5_PIc'

# Low-pass filter configuration
LPF_month_ALL = [120,0,24]

# detrend & deseason
detrend_covar = 1
detrend_MOC = 1
deseason_covar = 0
deseason_MOC = 0

use_resnet = 1 #dual-branch neural network (DBNN)

# Density-dependent MOC reconstruction
Full_depth_MOC = 0

# local neural network
use_local_inputs = 0
local_latind = 999  
#ACCESS_ESM1-5
# [0~172] for AMOC; local_latind = 119 for 25.5 north 
# [0~60] for SO abyssal MOC
# [0~22] for SO middepth MOC
#GFDL_CM4
# [0~490] for AMOC; local_latind = 119 for 25.5 north 

# Neural network hyperparameters
num_folds = 5
NNrepeats = 5
NN_neurons = [64,64]
loss_function = 'mse'
activation_function = 'leaky_relu'
epoch_max = 2000
batch_size = 600
reg_strength = 0.01
dropout_rate = 0.2
learning_rate = 0.001 #Adam default: 0.001
# Data handling parameters
use_input_branches = 0
use_scaler_y = 1
scaler_x_minmax = 0
scaler_x_robust = 0


# EOFs
PCA_variability_factor = 0 # ranges from 0 to 1
PCA_num = 0
PCA_y_variability_factor = 0 # ranges from 0 to 1
PCA_y_num = 0

# Layer-wise relevance propagation
LRP_rule = 'lrp.z'
LRP_plot = 0
# 'lrp.z', 'lrp.epsilon', 'lrp.w_square', 'lrp.flat', 'lrp.alpha_beta', 
# 'lrp.alpha_2_beta_1', 'lrp.alpha_2_beta_1_IB', 'lrp.alpha_1_beta_0', 'lrp.alpha_1_beta_0_IB', 
# 'lrp.z_plus', 'lrp.z_plus_fast', 'lrp.sequential_preset_a', 'lrp.sequential_preset_b', 
# 'lrp.sequential_preset_a_flat', 'lrp.sequential_preset_b_flat', 
# 'lrp.sequential_preset_b_flat_until_idx', 'deep_taylor', 'deep_taylor.bounded'
# LRP_rule = 'MyLRP'


#%% less important parameters
save_in_and_out = 1
lon_dec = 1  
density_smooth_MOC = 1
density_smooth_MOC_V2 = 1
cal_shap_value = 0
time_machine_range = 0  # year
tauu_xavg_online = 1

analysis_folder = os.path.join('E:', 'Analysis')
script_path_and_name =  os.path.join('D:\\OneDrive - University of California\\PythonWork\\CMIP6', 
                                     'NN_training_CMIP6_VX.py')
logger_name = 'logfile.log'
#%% Main function

def neural_network_training(MOC_id,covariate_names):
    
    #%% Define logger that can print useful information into a logfile
               
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(os.path.join(analysis_folder, logger_name))
    console_handler = logging.StreamHandler()

    # Set level and format for handlers
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Test the setup
    logger.info("logfile")

     
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logger.info("Using GPU for training.")
        logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    else:
        logger.info("Using CPU for training.") 

    logger.info(os.getenv('TF_GPU_ALLOCATOR'))
    
    # %% some flags
    use_time_machine = 1 if time_machine_range else 0
    use_PCA_x = 1 if PCA_variability_factor+PCA_num!=0 else 0
    if PCA_variability_factor != 0 and PCA_num != 0:
        raise ValueError("Both 'PCA_variability_factor' and 'PCA_num' cannot be non-zero simultaneously." 
                              "Please specify only one to avoid configuration conflicts.")
    use_PCA_y = 1 if PCA_y_variability_factor+PCA_y_num!=0 else 0
    if PCA_y_variability_factor != 0 and PCA_y_num != 0:
        raise ValueError("Both 'PCA_y_variability_factor' and 'PCA_y_num' cannot be non-zero simultaneously." 
                              "Please specify only one to avoid configuration conflicts.")
    use_LRP = 1 if LRP_rule else 0
    
    
    # %% directories
    def construct_data_directory():
        MOC_map = {1: 'AMOC', 2: 'SOMOC_middepth', 3: 'SOMOC_abyssal'}
        MOC_str = MOC_map.get(MOC_id, 'AMOC')
        
        deseadetrens_str = f"_Xd{'T' if detrend_covar else ''}{'S' if deseason_covar else ''}_Yd{'T' if detrend_MOC else ''}{'S' if deseason_MOC else ''}"
        lon_dec_str = '' if lon_dec == 1 else f'_LonDec{lon_dec}'
        
        if density_smooth_MOC:
            if density_smooth_MOC_V2 and MOC_id == 1 and CMIP_name == 'ACCESS_ESM1-5_PIc':
                density_smooth_MOC_str = '_rhoSmoothV2'
            else:
                density_smooth_MOC_str = '_rhoSmooth'
        else:
            density_smooth_MOC_str =''
        
        data_dir = os.path.join(analysis_folder, CMIP_name, 'BasinSpecific' + deseadetrens_str + density_smooth_MOC_str, MOC_str, lon_dec_str)

        return data_dir, MOC_str
    

    data_dir, MOC_str = construct_data_directory()
    
    
    # %% Main Execution -- data loading
    Grid = sio.loadmat(os.path.join(data_dir, 'Grid'))
    # load grids
    if CMIP_name == 'ACCESS_ESM1-5_PIc':
        Lat_gn = Grid['Lat_gn']
        Lon_gn = Grid['Lon_gn']
        lat_in=Lat_gn[0,:];  
        lon_in=Lon_gn[:,0]; 
        lat_psi=Lat_gn[11,:];  
        Nlons = lon_in.shape[0]
        Nlats = lat_in.shape[0]
        Nlats_psi = Nlats
    elif CMIP_name == 'GFDL_CM4_PIc':
        Lat_gn = Grid['Lat_gn']
        Lat_gr = Grid['Lat_gr']
        Lon_gr = Grid['Lon_gr']
        lat_in=Lat_gr[0,:];  
        lon_in=Lon_gr[:,0]; 
        lat_psi=Lat_gn[11,:];  
        Nlons = lon_in.shape[0]
        Nlats = lat_in.shape[0]
        Nlats_psi = lat_psi.shape[0]
    
    
    
    # load MOC strength
    if Full_depth_MOC:
        MatFN_MOC = os.path.join(data_dir, 'Psi_LatLevTime.mat')
        data_MOC = sio.loadmat(MatFN_MOC)
        rho2 = data_MOC['rho2_full']
        Psi = data_MOC['Psi_resi_full'].T # Time, Lev, Lat
        Nsamps = Psi.shape[0]
        Nlevs = Psi.shape[1]
        if MOC_id ==2:
            plt.figure(figsize=(10, 6))
        elif MOC_id ==1:
            plt.figure(figsize=(24, 6))
        elif MOC_id ==3:
            plt.figure(figsize=(18, 6))
        plt.pcolormesh(lat_psi,rho2,np.std(Psi/1e6,axis=0), cmap='RdYlBu_r', shading='auto')
        plt.gca().invert_yaxis()
        title = 'standard deviation of MOC (Sv)'
        plt.title(title)
        plt.xlabel('Latitude')
        plt.ylabel('Potential density')
        plt.colorbar(orientation='vertical')
        plt.savefig(os.path.join(data_dir, 'STD_FullDepth_MOC.png'),dpi=300)
        plt.show()
        
        Psi = Psi.reshape(Psi.shape[0],Psi.shape[1]*Psi.shape[2])
        Psi_mask = ~np.isnan(Psi).any(axis=0)
        Psi = Psi[:,Psi_mask]
            
    else:   
        MatFN_MOC = os.path.join(data_dir, 'Psi.mat')
        data_MOC = sio.loadmat(MatFN_MOC)
        Psi = data_MOC['Psi'].T
        
    Nsamps = Psi.shape[0]
    del data_MOC
    
    if covariate_names == 'obp' and not Full_depth_MOC:
        temp2 = np.std(Psi,axis=0)
        plt.figure(figsize=(6, 4))
        plt.plot(lat_in,temp2.T)
        title = 'standard deviation of MOC'
        plt.title(title)
        plt.xlabel('Latitude')
        plt.ylabel('Standard deviation')
        plt.grid()
        plt.savefig(os.path.join(data_dir, 'STD_MOC.png'),dpi=300)
        plt.show()
        
    if use_local_inputs:
        Psi = Psi[:,local_latind].reshape(-1,1)
        
        if CMIP_name == 'GFDL_CM4_PIc':
            local_latind_in = np.argmin( np.abs(lat_in - lat_psi[local_latind]))
            logger.info(f'Using inputs only at latitude {lat_in[local_latind_in]:.2f}')
            logger.info(f'Using outputs only at latitude {lat_psi[local_latind]:.2f}')
        else:
            local_latind_in = local_latind
            logger.info(f'Using inputs and outputs only at latitude {lat_psi[local_latind]:.2f}')
        
        
    # load covariates (i.e., input features)
    CovariateNumInd = np.empty((0))
    CovariatALL = np.empty((Nsamps,0))
    # remove covariates that are not used
    if len(covariate_names) != 0: 
        for name in covariate_names.split(','):
            if name !='zws' or tauu_xavg_online!=1:
                temp = sio.loadmat(os.path.join(data_dir, name))[name]
                temp = np.squeeze(temp.T) # time lat lon
                logger.info(f'loaded {"zonally-varying" if temp.ndim == 3 else "zonally-averaged"} inputs: {name}')
                if temp.ndim == 3:
                    if "," not in covariate_names:
                        temp2 = np.log10(np.std(temp,axis=0))
                        max_val = np.nanmax(temp2)
                        # Create a symmetric colormap around zero
                        norm = Normalize(vmin=max_val-2, vmax=max_val)
                        if MOC_id ==2:
                            plt.figure(figsize=(18, 4))
                        elif MOC_id ==1:
                            plt.figure(figsize=(12, 9))
                        elif MOC_id ==3:
                            plt.figure(figsize=(18, 6))
                        plt.pcolormesh(lon_in,lat_in,temp2, cmap='RdYlBu_r', norm=norm, shading='auto')
                        title = 'standard deviation of ' + covariate_names+ ' (log10)'
                        plt.title(title)
                        plt.xlabel('Longitude')
                        plt.ylabel('Latitude')
                        plt.colorbar(orientation='vertical')
                        plt.savefig(os.path.join(data_dir, 'STD_' + covariate_names  + '.png'),dpi=300)
                        plt.show()
                if temp.ndim == 2:
                    if "," not in covariate_names:
                        temp2 = np.std(temp,axis=0)
                        plt.figure(figsize=(6, 4))
                        plt.plot(lat_in,temp2.T)
                        title = 'standard deviation of ' + covariate_names
                        plt.title(title)
                        plt.xlabel('Latitude')
                        plt.ylabel('Standard deviation')
                        plt.grid()
                        plt.savefig(os.path.join(data_dir, 'STD_' + covariate_names  + '.png'),dpi=300)
                        plt.show()
                if use_local_inputs:
                    temp = temp[:, local_latind_in]
                    if len(temp.shape)==1:
                        temp = temp.reshape(-1,1) 
                if temp.ndim == 3:
                    temp = temp.reshape(temp.shape[0],temp.shape[1]*temp.shape[2])
                    
                if name!='zws_xavg':
                    land_mask = ~np.isnan(temp).any(axis=0)
                    temp = temp[:,land_mask]
                
                    
                CovariatALL = np.concatenate((CovariatALL,temp),axis = 1)
                CovariateNumInd = np.append(CovariateNumInd,temp.shape[1])
                del temp
    else:
        logger.info('Error: found wrong input names')
        
        
    # load zonal wind stress and take zonal average
    if "zws"in covariate_names and tauu_xavg_online:
        name='zws'
        temp = sio.loadmat(os.path.join(data_dir, name))[name]
        temp = np.squeeze(temp.T) # time lat lon
        land_mask = ~np.isnan(temp).any(axis=0)
        temp = np.nanmean(temp,axis=2) # zonal average
        if use_local_inputs:
            temp = temp[:, local_latind_in].reshape(-1,1)
        logger.info('Calculate zonally averaged zws based on loaded zoanlly-varying zws')
        CovariatALL = np.concatenate((CovariatALL,temp),axis = 1)
        CovariateNumInd = np.append(CovariateNumInd,temp.shape[1])
        names_list = covariate_names.split(',')
        names_list = [name + '_xavg' if name == 'zws' else name for name in names_list]
        covariate_names = ','.join(names_list)


    
    # use sea surface density information back in time 
    if use_time_machine:
        # use zonal-averaged SST time series at differnet latitude back in time
        temp = sio.loadmat(os.path.join(data_dir, 'ssd'))['ssd']
        # ssd_xavg(latitude, months)
        ssd_xavg = np.nanmean(temp[:, :, :], axis=0)    
        # Example data dimensions
        n_years = int(Nsamps / 12)  # Total years
        # Reshape data to (latitude, 12, years)
        ssd_xavg = ssd_xavg.reshape(Nlats, 12, n_years,order='F')
        
        # Calculate yearly mean for each latitude
        ssd_history =  np.squeeze(np.mean(ssd_xavg, axis=1)).T
        # plt.plot(ssd_history[0:200,0],'-')
        
        output_months = Nsamps - time_machine_range*12
        historical_ssd = np.empty((time_machine_range, Nlats , output_months))
        for month in range(output_months):
            start_year = month//12 
            end_year = start_year + time_machine_range
            historical_ssd[:, :, month] = ssd_history[start_year:end_year,:]
    
        historical_data = historical_ssd.reshape(historical_ssd.shape[0]*historical_ssd.shape[1],
                                                  historical_ssd.shape[2],order='F').T
        
        del historical_ssd
    
        Psi = Psi[time_machine_range*12:,:]
        CovariatALL = CovariatALL[time_machine_range*12:,:]
        
        CovariatALL = np.concatenate((CovariatALL,historical_data),axis = 1)
        CovariateNumInd = np.append(CovariateNumInd,historical_data.shape[1])    
        Nsamps = Nsamps-time_machine_range*12
        logger.info(f"!!!use previous {time_machine_range} year zonally-averaged SSD!!!")
    
    
    
    logger.info(f"number of samples in time: {Nsamps}")
    logger.info(f"number of input quantities: {len(CovariateNumInd)}")
    logger.info(f"feature number of each input quantity: {CovariateNumInd}")
    logger.info(f"feature number of output quantity: {Psi.shape[1]}")
    
    
    
    
    Years = np.arange(1, Nsamps + 1) / 12
    CovariateNumIndCum = np.cumsum(CovariateNumInd)
    CovariateNumIndCum = np.insert(CovariateNumIndCum,0,0)
    CovariateNumIndCum  = CovariateNumIndCum.astype(int)
    CovariateNumInd  = CovariateNumInd.astype(int)
    
    
    Psi_raw = Psi.copy()
    CovariatALL_raw = CovariatALL.copy()
    
    
    
    #%% Low pass filter loop starts here
    for LPF_month in LPF_month_ALL: #0,24,120
    
        if LPF_month!=0:
            logger.info('------------------------------------------------------------------------')
            logger.info('------------------------------------------------------------------------')
            logger.info(f'Applying {LPF_month/12}-year low pass filter ...')
            logger.info('------------------------------------------------------------------------')
            logger.info('------------------------------------------------------------------------')
            
        use_LPF_y = 1 if LPF_month else 0
        use_LPF_x = 1 if LPF_month else 0
        
        
        def create_NN_name():
            NN_name = 'MLP'  # Default name
            
            if use_local_inputs:
                NN_name = 'LocalMLP'
            elif use_resnet:
                NN_name = 'ResNet'
            elif use_time_machine:
                NN_name = f'TMMLP_SSD_{time_machine_range}Years'

                
            if use_PCA_x or use_PCA_y and not Full_depth_MOC:
                if use_PCA_y and use_PCA_x:
                    NN_name = NN_name + f'PCAinX{PCA_num + PCA_variability_factor}Y{PCA_y_num + PCA_y_variability_factor}'
                elif use_PCA_y:
                    NN_name = NN_name + f'PCAinY{PCA_y_num + PCA_y_variability_factor}'           
                else:      
                    NN_name = NN_name + f'PCAinX{PCA_num + PCA_variability_factor}'
                
            if Full_depth_MOC:
                if use_PCA_y:
                    if use_PCA_x:
                        NN_name = f'FullDepth_PCAinX{PCA_num + PCA_variability_factor}Y{PCA_y_num + PCA_y_variability_factor}'+NN_name
                    else:
                        NN_name = f'FullDepth_PCAinY{PCA_y_num + PCA_y_variability_factor}'+NN_name
                else:
                    NN_name = 'FullDepth_'+NN_name

            activation_function_str = '_'+activation_function+'Activation' if activation_function !='leaky_relu' else ''
            loss_function_str = '_'+str(loss_function)+'loss' if loss_function!='mse' else ''
            scaler_str = 'NoYScaler_' if not use_scaler_y else ''
            scaler_str = scaler_str+'MinMaxXScaler_' if scaler_x_minmax else scaler_str
            scaler_str = scaler_str+'RobustXScaler_' if scaler_x_robust else scaler_str
            LPF_str = f'_LPF{int(LPF_month / 12)}Year' if use_LPF_x else ''
            NN_structure_str = 'x'.join(map(str, NN_neurons)) + ('useBranches' if use_input_branches else '')
            reg_str = f'Reg{reg_strength}' + (f'Drop{dropout_rate}' if dropout_rate != 0 else '')
            batch_size_str =  f'BS{batch_size}_' if batch_size !=600 else ''
            interp_str = f'{LRP_rule}_' if use_LRP else ''
            
            return f"{interp_str}{NN_name}_{scaler_str}Neur{NN_structure_str}_{batch_size_str}{num_folds}foldCV_{reg_str}{loss_function_str}{activation_function_str}{LPF_str}",LPF_str
        NN_name,LPF_str = create_NN_name()
        
        def construct_output_directory(data_dir, NN_name):
            flat_names_list = covariate_names.split(",")
            covariate_names_str = "+".join(flat_names_list)   
            if use_local_inputs:
                Lat_str = f'{lat_psi[local_latind]:.2f}'
                output_dir = os.path.join(data_dir, 'results', NN_name, Lat_str,covariate_names_str)
            else:
                output_dir = os.path.join(data_dir, 'results', NN_name, covariate_names_str)
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output path: {'created successfully' if os.path.exists(output_dir) else 'already exists'}")
            logger.info(f"The output path is {output_dir}")
            return output_dir, covariate_names_str,flat_names_list
        output_dir, covariate_names_str,flat_names_list = construct_output_directory(data_dir, NN_name)
        
        
        
        
        #%% Apply low pass filter if needed   
        
        def apply_low_pass_filter(data, cutoff_freq, order=5, sampling_rate=1, padding_length=None):
            """Applies a Butterworth low-pass filter to the given data."""
            sos = butter(order, cutoff_freq, btype='low', output='sos', analog=False, fs=sampling_rate)
            
            # Apply Boundary Padding to the data before filtering
            padded_data = np.pad(data, [(padding_length, padding_length), (0, 0)], mode='reflect')
            
            # Apply the filter across each column without explicit looping
            filtered_data = sosfilt(sos, padded_data, axis=0)
            
            # Remove padding
            return filtered_data[padding_length:-padding_length]
        
        if use_LPF_y or use_LPF_x:
            # Calculate the sampling rate (monthly data)
            sampling_rate = 1  # Data is sampled monthly    
            cutoff_freq = 1 / LPF_month
            order = 5
            padding_length = 2 * LPF_month
        
        if use_LPF_y:
            Psi = apply_low_pass_filter(Psi_raw, cutoff_freq, order, sampling_rate, padding_length)               
            if covariate_names == 'obp':  
                plt.figure(figsize=(12, 9))
                plt.subplot(3, 1, 1)
                plt.plot(Years, Psi_raw[:,0], color='blue', label='Original monthly data')
                plt.plot(Years, Psi[:,0], color='black', label='LPF-'+str(LPF_month/12)+'Year')
                plt.ylabel('MOC')
                plt.title('Original and low-passed MOC strength')
                plt.legend()
                plt.subplots_adjust(hspace=0.4)
                plt.subplot(3, 1, 2)
                plt.plot(Years[:600], Psi_raw[:600,0], color='blue', label='Original monthly data')
                plt.plot(Years[:600], Psi[:600,0], color='black', label='LPF-'+str(LPF_month/12)+'Year')
                plt.ylabel('MOC')
                plt.title('Zoom in to the first 50 years')
                plt.subplot(3, 1, 3)
                plt.plot(Years[-600:], Psi_raw[-600:,0], color='blue', label='Original monthly data')
                plt.plot(Years[-600:], Psi[-600:,0], color='black', label='LPF-'+str(LPF_month/12)+'Year')
                plt.xlabel('Years')
                plt.ylabel('MOC')
                plt.title('Zoom in to the last 50 years')
                plt.legend()
                plt.savefig(os.path.join(output_dir, 'LPF_Psi.png'),dpi=300)
        else:
            Psi = Psi_raw.copy()
          
                
        if use_LPF_x:        
            CovariatALL = apply_low_pass_filter(CovariatALL_raw, cutoff_freq, order, sampling_rate, padding_length)  
            if len(CovariateNumInd) ==1:      
                plt.figure(figsize=(12, 9))
                plt.subplot(3, 1, 1)
                plt.plot(Years, CovariatALL_raw[:, 0], color='blue', label='Original monthly data')
                plt.plot(Years, CovariatALL[:, 0], color='black', label='LPF-'+str(LPF_month/12)+'Year')
                plt.ylabel('CoVariate')
                plt.title('Original and low-passed CoVariate - first')
                plt.subplots_adjust(hspace=0.4)
                plt.subplot(3, 1, 2)
                plt.plot(Years[:600], CovariatALL_raw[:600, 0], color='blue', label='Original monthly data')
                plt.plot(Years[:600], CovariatALL[:600, 0], color='black', label='LPF-'+str(LPF_month/12)+'Year')
                plt.ylabel('CoVariate')
                plt.title('Zoom in to the first 50 years')   
                plt.subplot(3, 1, 3)
                plt.plot(Years[-600:], CovariatALL_raw[-600:, 0], color='blue', label='Original monthly data')
                plt.plot(Years[-600:], CovariatALL[-600:, 0], color='black', label='LPF-'+str(LPF_month/12)+'Year')
                plt.xlabel('Years')
                plt.ylabel('CoVariate')
                plt.title('Zoom in to the last 50 years')       
                plt.legend()
                plt.savefig(os.path.join(output_dir, 'LPF_'+ covariate_names_str  +'_westernmost.png'),dpi=300)
                
                plt.figure(figsize=(12, 9))
                plt.subplot(3, 1, 1)
                plt.plot(Years, CovariatALL_raw[:, -1], color='blue', label='Original monthly data')
                plt.plot(Years, CovariatALL[:, -1], color='black', label='LPF-'+str(LPF_month/12)+'Year')
                plt.ylabel('CoVariate')
                plt.title('Original and low-passed CoVariate - last')
                plt.subplots_adjust(hspace=0.4)
                plt.subplot(3, 1, 2)
                plt.plot(Years[:600], CovariatALL_raw[:600, -1], color='blue', label='Original monthly data')
                plt.plot(Years[:600], CovariatALL[:600, -1], color='black', label='LPF-'+str(LPF_month/12)+'Year')
                plt.ylabel('CoVariate')
                plt.title('Zoom in to the first 50 years')   
                plt.subplot(3, 1, 3)
                plt.plot(Years[-600:], CovariatALL_raw[-600:, -1], color='blue', label='Original monthly data')
                plt.plot(Years[-600:], CovariatALL[-600:, -1], color='black', label='LPF-'+str(LPF_month/12)+'Year')
                plt.xlabel('Years')
                plt.ylabel('CoVariate')
                plt.title('Zoom in to the last 50 years')       
                plt.legend()
                plt.savefig(os.path.join(output_dir, 'LPF_'+ covariate_names_str  +'_easternmost.png'),dpi=300)  
                
                plt.figure(figsize=(12, 9))
                plt.subplot(3, 1, 1)
                plt.plot(Years, CovariatALL_raw[:, CovariatALL.shape[1]//2], color='blue', label='Original monthly data')
                plt.plot(Years, CovariatALL[:, CovariatALL.shape[1]//2], color='black', label='LPF-'+str(LPF_month/12)+'Year')
                plt.ylabel('CoVariate')
                plt.title('Original and low-passed CoVariate - middle')
                plt.subplots_adjust(hspace=0.4)
                plt.subplot(3, 1, 2)
                plt.plot(Years[:600], CovariatALL_raw[:600, CovariatALL.shape[1]//2], color='blue', label='Original monthly data')
                plt.plot(Years[:600], CovariatALL[:600, CovariatALL.shape[1]//2], color='black', label='LPF-'+str(LPF_month/12)+'Year')
                plt.ylabel('CoVariate')
                plt.title('Zoom in to the first 50 years')   
                plt.subplot(3, 1, 3)
                plt.plot(Years[-600:], CovariatALL_raw[-600:, CovariatALL.shape[1]//2], color='blue', label='Original monthly data')
                plt.plot(Years[-600:], CovariatALL[-600:, CovariatALL.shape[1]//2], color='black', label='LPF-'+str(LPF_month/12)+'Year')
                plt.xlabel('Years')
                plt.ylabel('CoVariate')
                plt.title('Zoom in to the last 50 years')       
                plt.legend()
                plt.savefig(os.path.join(output_dir, 'LPF_'+ covariate_names_str  +'_middle.png'),dpi=300)    
        else:
            CovariatALL = CovariatALL_raw.copy()
        
        #%% Define the neural network
        
        # Make another logfile to store loss during training
        log_path =os.path.join(output_dir, 'training_logs.txt')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        class PrintTrainingOnTextEvery10EpochsCallback(Callback):
            def __init__(self, log_path):
                super().__init__()
                self.log_path = log_path
        
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 10 == 0:  # Log every 10 epochs
                    with open(self.log_path, "a") as log_file:
                        log_file.write(
                            f"Epoch: {epoch:>3} | "
                            f"Loss: {logs.get('loss', 0):.2e} | "
        #                        f"Accuracy: {logs.get('accuracy', 0):.2e} | "
                            f"Validation loss: {logs.get('val_loss', 0):.2e} |\n "
        #                        f"Validation accuracy: {logs.get('val_accuracy', 0):.2e}\n"
                        )
                        print(
                            f"Epoch {epoch:>3} - "
                            f"Loss: {logs.get('loss', 0):.2e}, "
        #                        f"Accuracy: {logs.get('accuracy', 0):.2e}, "
                            f"Validation loss: {logs.get('val_loss', 0):.2e}, "
        #                        f"Validation accuracy: {logs.get('val_accuracy', 0):.2e}"
                        )
        
        my_callbacks = [
            PrintTrainingOnTextEvery10EpochsCallback(log_path=log_path),
        ]   
        
        def get_activation(activation_function):
            if activation_function == 'leaky_relu':
                return LeakyReLU(alpha=0.2)
            elif activation_function == 'relu':
                return 'relu'
            elif activation_function == 'sigmoid':
                return 'sigmoid'
            elif activation_function == 'tanh':
                return 'tanh'
            elif activation_function == 'elu':
                return 'elu'
            elif activation_function == 'linear':
                return 'linear'
            else:
                raise ValueError(f"Unsupported activation function: {activation_function}")


        def gpu_memory():
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            memory_info = memory_info['current'] / (1024 ** 2)# Convert from bytes to MiB
            logging.info(f'TensorFlow memory usage: {memory_info:.2f} MiB')
            print(f'TensorFlow memory usage: {memory_info:.2f} MiB')
            return  memory_info 
        
        
        
        #######################################################################
        #                       define the neural network                     #
        #######################################################################
        
        def train_model(X_train, y_train, X_test, y_test,scaler_y):
            
            activation = get_activation(activation_function)
            # Create a new model with random initial weights and biases
            inputs = tf.keras.Input(shape=(X_train.shape[1],))
            output1 = Dense(units=NN_neurons[0],
                               kernel_regularizer=regularizers.l2(reg_strength))(inputs)
            output = Activation(activation)(output1)
            if dropout_rate!=0:
                output = Dropout(dropout_rate)(output)

            # Add more dense layers with non-linear activation 
            if len(NN_neurons) > 1:
                for i in range(1, len(NN_neurons)):
                    output = Dense(units=NN_neurons[i], activation=activation,
                                   kernel_regularizer=regularizers.l2(reg_strength))(output)
                    if dropout_rate!=0:
                        output = Dropout(dropout_rate)(output)
                        
            
            if use_resnet:
                if dropout_rate!=0:
                    output_skip = Dropout(dropout_rate)(output1)                  
                output_skip = Dense(units=y_train.shape[1], activation='linear',
                                   kernel_regularizer=regularizers.l2(reg_strength))(output_skip)
                               
                output = Dense(units=y_train.shape[1], activation='linear')(output)
                
                output = Add(name='add_layer')([output,output_skip])
            else:
                # Add the final linear output layer
                output = Dense(units=y_train.shape[1], activation='linear')(output)
            
            
            # Create the model
            model = Model(inputs=inputs, outputs=output)
            
            if fold_no + ens_no == 2:
                # show the model summary
                model.summary()
                with open(os.path.join(output_dir, 'model_' + covariate_names_str  + '.txt'), 'w') as f:
                    model.summary(print_fn=lambda x: f.write(x + '\n'))
                
                dot_img_file = os.path.join(output_dir, 'model_' + covariate_names_str  + '.png')
                plot_model(model, to_file=dot_img_file,
                           show_shapes=True,
                           show_dtype=False,
                           show_layer_names=True,
                           rankdir='LR',
                           expand_nested=False,
                           dpi=300,
                           show_layer_activations=True)
                
            # Compile the model with mean squared error loss and Adam optimizer
            model.compile(loss=loss_function, optimizer=Adam(learning_rate=learning_rate))
            
            # Define early stopping callback
            early_stopping = EarlyStopping(patience=50, monitor='val_loss', mode='min', restore_best_weights=1, verbose=1)
            gpu_memory()
            # Train the model with early stopping callback
            history = model.fit(X_train, y_train, epochs=epoch_max, batch_size=batch_size,
                                validation_data=(X_test, y_test), callbacks=[early_stopping,my_callbacks],verbose=0)
            gpu_memory()
            # Evaluate the performance on the testing set (less useful)
            skill = model.evaluate(X_test, y_test, verbose=0) 
            
            
            
###############################################################################
#                        codes for explainable AI                             #
###############################################################################
            if cal_shap_value:
                # Create a smaller background dataset for SHAP
                background = X_train[np.random.choice(X_train.shape[0], 1000, replace=False)]
                
                # Create a SHAP DeepExplainer
                explainer = shap.DeepExplainer(model, background)
                
                # Function to explain predictions in batches
                
                def compute_shap_values_in_batches(explainer, data, batch_size=100):
                    shap_values = []
                    for i in range(0, len(data), batch_size):
                        print(f'i = {i}')
                        batch_data = data[i:i+batch_size]
                        batch_shap_values = explainer.shap_values(batch_data)
                        shap_values.append(batch_shap_values)
                    return np.concatenate(shap_values, axis=0)
                # Choose some instances to explain
                X_to_explain = X_train[:100]  # explaining the first 100 instances for example
                
                # Compute SHAP values in batches
                shap_values = compute_shap_values_in_batches(explainer, X_to_explain, batch_size=100)
                
                # Visualize the SHAP values for the first output feature
                shap.summary_plot(shap_values[:,:,0], X_to_explain)
                shap.summary_plot(shap_values[:,:,170], X_to_explain, plot_type="bar")

                # Convert shap_values to Explanation object if needed
                
                
            if LRP_rule == 'MyLRP':
                
                intv = 3 if MOC_id==1 else 1
                def lrp_dense(model, R_output, layer_idx, X_in,output_idx, epsilon):
                    """
                    Apply LRP for a Dense layer.
                    Args:
                    - model: The Keras model.
                    - R_output: Relevance scores from the higher layer.
                    - layer_idx: Index of the current layer.
                    - X: Input data.
                    - epsilon: Stabilization term to avoid division by zero.
                    """
                    W = model.layers[layer_idx].get_weights()[0]  # Weights of the current layer
                    b = model.layers[layer_idx].get_weights()[1]  # Biases of the current layer
                    
                    if model.layers[-1] == model.layers[layer_idx]:
                        W = W[:,output_idx]
                        W = np.expand_dims(W, axis=1) 
                        b = b[output_idx]
                    if layer_idx == 1:  # Special case for the first layer
                         A_prev = X_in
                    else:
                         prev_layer = tf.keras.Model(inputs=model.input, outputs=model.layers[layer_idx - 1].output)
                         A_prev = prev_layer.predict(X_in)
                    
                    Zj = np.dot(A_prev, W) + b
                    # Add stabilization term
                    Zj += epsilon * np.where(Zj >= 0, 1, -1)
                    
                    R_prev = A_prev * (np.dot(R_output/Zj, W.T) )
                    
                    return R_prev

                
                def lrp(model, X_in, R_initial, epsilon=0):
                    """
                    Apply LRP to the entire model.
                    Args:
                    - model: The Keras model.
                    - X: Input data.
                    - R_initial: Initial relevance scores (typically the model's output).
                    - epsilon: Stabilization term to avoid division by zero.
                    """
                    R = R_initial  # Initial shape: (12000,172)
                    relevances = np.empty((R_initial.shape[1],X_in.shape[1]))
                    absolute_relevances = np.empty((R_initial.shape[1],X_in.shape[1]))
                    for output_idx in range(0,y_test.shape[1],intv):  # Iterate over each output feature
                        print(output_idx)
                        R_output = R[:, output_idx]  # Shape: (12000,)
                        R_output = np.expand_dims(R_output, axis=1)  # Shape: (12000, 1)
                        for layer_idx in reversed(range(1, len(model.layers))):  # Start from the last layer and exclude input layer
                            # print(layer_idx)
                            if isinstance(model.layers[layer_idx], tf.keras.layers.Dense):
                                R_output = lrp_dense(model, R_output, layer_idx, X_in, output_idx, epsilon)  # Shape changes layer by layer
                            elif isinstance(model.layers[layer_idx], tf.keras.layers.Dropout):
                                continue  # Skip dropout layers
                            else:
                                raise NotImplementedError("LRP not implemented for this layer type.")
                                
                        
                        relevances[output_idx,:] = np.mean(R_output, axis=0).T
                        absolute_relevances[output_idx,:] = np.mean(np.abs(R_output), axis=0).T
                    return relevances,absolute_relevances
                
                
                
                mean_relevances,mean_absolute_relevances = lrp(model, X_test, R_initial=model.predict(X_test))  # Shape: (12000, 10337)
                
                
            elif use_LRP:
                
                # Calculate relevance of input considering MOC at all latitudes
                analyzer = innvestigate.create_analyzer(LRP_rule, model)
                analysis = analyzer.analyze(X_test)
                
                mean_relevance= np.mean(analysis, axis=0)
                mean_absolute_relevance= np.mean(abs(analysis), axis=0)
                
                os.makedirs(os.path.join(output_dir,'LRP_fold'+str(fold_no)+'_ens'+str(ens_no)), exist_ok=True)
                sio.savemat(os.path.join(output_dir,'LRP_fold'+str(fold_no)+'_ens'+str(ens_no),'DomainwideRelevance.mat'), 
                            {'mean_relevance': mean_relevance, 'mean_absolute_relevance': mean_absolute_relevance, 
                             'land_mask': land_mask, 'Nlats':Nlats, 'Nlons':Nlons,
                              'CovariateNumIndCum':CovariateNumIndCum,'covariate_names':covariate_names})

                # Calculate relevance of input only considering MOC at each latitudes
                intv = 1
                relevances_std = np.empty((y_test.shape[1],X_test.shape[1]))
                relevances_abs_std = np.empty((y_test.shape[1],X_test.shape[1]))
                mean_relevances = np.empty((y_test.shape[1],X_test.shape[1]))
                mean_absolute_relevances = np.empty((y_test.shape[1],X_test.shape[1]))
                relevances_sum = np.empty((y_test.shape[1],y_test.shape[0]))
                # Loop through each output feature
                for i in range(0,y_test.shape[1],intv):
                    # Analyze relevance for the i-th output feature
                    analyzer_output_specific = innvestigate.create_analyzer(LRP_rule, model, neuron_selection_mode="index")
                    relevance_scores = analyzer_output_specific.analyze(X_test, i)  # Relevance for the i-th output feature
                    
                                        
                    relevances_std[i,:] = np.std(relevance_scores, axis=0)   
                    relevances_abs_std[i,:] = np.std(abs(relevance_scores), axis=0)   
                    mean_relevances[i,:] = np.mean(relevance_scores, axis=0)  # Average over all samples
                    mean_absolute_relevances[i,:] = np.mean(abs(relevance_scores), axis=0)  # Average over all samples
                    relevances_sum[i,:] = np.sum(relevance_scores, axis=1)
                    
                    
                    
            # ploting LRP results
            if use_LRP:
                os.makedirs(os.path.join(output_dir,'LRP_fold'+str(fold_no)+'_ens'+str(ens_no)), exist_ok=True)
                os.makedirs(os.path.join(output_dir,'LRP_abs_fold'+str(fold_no)+'_ens'+str(ens_no)), exist_ok=True)
                sio.savemat(os.path.join(output_dir,'LRP_fold'+str(fold_no)+'_ens'+str(ens_no),'LatwideRelevance.mat'), 
                            {'mean_relevances': mean_relevances, 'mean_absolute_relevances': mean_absolute_relevances,
                             'relevances_std':relevances_std,'relevances_abs_std':relevances_abs_std,'relevances_sum':relevances_sum,
                             'land_mask': land_mask, 'Nlats':Nlats, 'Nlons':Nlons,
                             'OutFeatureNum':y_train.shape[1], 'intv':intv,
                            'CovariateNumIndCum':CovariateNumIndCum,'covariate_names':covariate_names})
                
            if use_LRP and LRP_plot:
                ind = 0
                for name in covariate_names.split(','):
                    ind = ind + 1
                    n = CovariateNumIndCum
                    temp = mean_relevances[:,n[ind-1]:n[ind]]
                    temp2 = mean_absolute_relevances[:,n[ind-1]:n[ind]]
                    if name!= 'zws_xavg':        
                        mean_relevances_yz = np.full((y_train.shape[1],Nlats*Nlons), np.nan)  # Initialize with NaN
                        mean_relevances_yz[:,land_mask] = temp  # Only fill valid locations
                        mean_relevances_yz = mean_relevances_yz.reshape((y_test.shape[1],Nlats, Nlons))
                        
                        mean_absolute_relevances_yz = np.full((y_train.shape[1],Nlats*Nlons), np.nan)  # Initialize with NaN
                        mean_absolute_relevances_yz[:,land_mask] = temp2  # Only fill valid locations
                        mean_absolute_relevances_yz = mean_absolute_relevances_yz.reshape((y_test.shape[1],Nlats, Nlons))
                        
                        
                        # plot the spatial pattern of relevance score
                        for i in range(0,y_test.shape[1],intv):
                            if MOC_id ==2:
                                plt.figure(figsize=(15, 3))
                                vlim = 0.002
                            elif MOC_id ==1:
                                plt.figure(figsize=(10, 8))
                                vlim = 0.002
                            elif MOC_id ==3:
                                plt.figure(figsize=(15, 6))
                                vlim = 0.001
                                
                            plt.title(f"{name}; Mean Relevance Score for MOC at latitude {lat_psi[i]:.1f}")
                            plt.pcolormesh(lon_in,lat_in,mean_relevances_yz[i], cmap='RdBu_r',vmin=-vlim, vmax=vlim)  # Adjust reshape as necessary
                            plt.colorbar()
                            plt.plot([lon_in[0],lon_in[-1]],[lat_psi[i],lat_psi[i]],'--k')
                            plt.xlabel('Longitude')
                            plt.ylabel('Latitude')
                            plt.savefig(os.path.join(output_dir, 'LRP_fold'+str(fold_no)+'_ens'+str(ens_no), name  +'_LatwideRelevance_' + str(i) + '.png'),dpi=300)
                            plt.show()
                            
                            
                        for i in range(0,y_test.shape[1],intv):
                            # plot the spatial pattern of absolute relevance score
                            if MOC_id ==2:
                                plt.figure(figsize=(15, 3))
                                vlim = 0.004
                            elif MOC_id ==1:
                                plt.figure(figsize=(10, 8))
                                vlim = 0.004
                            elif MOC_id ==3:
                                plt.figure(figsize=(15, 6))
                                vlim = 0.002
                                
                                
                            plt.title(f"{name}; Mean Absolute Relevance Score for MOC at latitude {lat_psi[i]:.1f}")
                            plt.pcolormesh(lon_in,lat_in,mean_absolute_relevances_yz[i], cmap='RdYlBu_r',vmin=0, vmax=vlim)  # Adjust reshape as necessary
                            plt.colorbar()
                            plt.plot([lon_in[0],lon_in[-1]],[lat_psi[i],lat_psi[i]],'--k')
                            plt.xlabel('Longitude')
                            plt.ylabel('Latitude')
                            plt.savefig(os.path.join(output_dir, 'LRP_abs_fold'+str(fold_no)+'_ens'+str(ens_no), name  +'_LatwideRelevance_' + str(i) + '.png'),dpi=300)
                            plt.show()
                            
                            
                            
                    else:
                        for i in range(0,y_test.shape[1],intv):
                            
                            # plot relevance score of zonal averaged quantity
                            if MOC_id ==2:
                                plt.figure(figsize=(10, 8))
                                vlim = 0.002
                            elif MOC_id ==1:
                                plt.figure(figsize=(12, 6))
                                vlim = 0.001
                            elif MOC_id ==3:
                                plt.figure(figsize=(10, 8))
                                vlim = 0.001
                            plt.title(f"{name}; latitude {lat_psi[i]:.1f}")
                            plt.plot(lat_in,temp[i,:])  
                            plt.xlabel('Latitude')
                            plt.ylabel('Mean Relevance Score')
                            plt.ylim([-vlim ,vlim])
                            y1, y2 = plt.ylim()
                            plt.plot([lat_psi[i],lat_psi[i]],[y1, y2],'--k')
                            plt.savefig(os.path.join(output_dir, 'LRP_fold'+str(fold_no)+'_ens'+str(ens_no), name  +'_LatwideRelevance_' + str(i) + '.png'),dpi=300)
                            plt.show()
                            
                            
                        for i in range(0,y_test.shape[1],intv):
                            # plot absolute relevance score of zonal averaged quantity
                            if MOC_id ==2:
                                vlim = 0.008
                                plt.figure(figsize=(10, 8))
                            elif MOC_id ==1:
                                plt.figure(figsize=(12, 6))
                                vlim = 0.004
                            elif MOC_id ==3:
                                plt.figure(figsize=(10, 8))
                                vlim = 0.004
                                
                            plt.title(f"{name}; latitude {lat_psi[i]:.1f}")
                            plt.plot(lat_in,temp2[i,:])  
                            plt.xlabel('Latitude')
                            plt.ylabel('Mean Absolute Relevance Score')
                            plt.ylim([0, vlim])
                            y1, y2 = plt.ylim()
                            plt.plot([lat_psi[i],lat_psi[i]],[y1, y2],'--k')
                            plt.savefig(os.path.join(output_dir, 'LRP_abs_fold'+str(fold_no)+'_ens'+str(ens_no), name  +'_LatwideRelevance_' + str(i) + '.png'),dpi=300)
                            plt.show()
                            
###############################################################################
#                     codes for explainable AI ends                           #
###############################################################################

                
            # Make the prediction with the actual scale in the testing set
            if use_PCA_y:
                pred = model.predict(X_test)
            else:
                pred = scaler_y.inverse_transform(model.predict(X_test))
            
            # Calculate the R2 value in the testing set at each latitude 
            R2_AllLats = []
            if use_PCA_y:
                truth = y_test
            else:
                truth = scaler_y.inverse_transform(y_test)
            for latind in range(y_test.shape[1]):
                y_lat = truth[:, latind];
                y_pred_lat = pred[:, latind];
                R2_AllLats.append(r2_score(y_lat, y_pred_lat))
                
            model.save(os.path.join(output_dir,'model_fold'+str(fold_no)+'_ens'+str(ens_no)+'.h5'))

            return skill, history, pred, R2_AllLats
        
        
        #%% 
        #######################################################################
        #                             Train loop                              #
        #######################################################################
        X = CovariatALL
        y = Psi/1e6
        # if not os.path.exists(os.path.join(output_dir, 'inputs_info.mat')):
        np.save(os.path.join(output_dir, 'intputs.npy'), X)
        np.save(os.path.join(output_dir, 'outputs.npy'), y)
        sio.savemat(os.path.join(output_dir,'inputs_info.mat'),
                    {'land_mask': land_mask, 'Nlats':Nlats, 'Nlons':Nlons, 'Nsamps':Nsamps,
                     'CovariateNumIndCum':CovariateNumIndCum,
                     'covariate_names':covariate_names})
        np.save(os.path.join(output_dir, 'land_mask.npy'), land_mask)
                
        trained_models = []
        y_pred_reconstructed_allfolds = []
        scaler_y_allfolds = []
        
        # Train the neural network multiple times using k-fold cross validation
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=num_folds, shuffle=False)
        fold_no = 1
        for trainind, testind in kfold.split(X, y):   
            # break
            X_train = X[trainind,:]
            y_train = y[trainind,:]
            X_test = X[testind,:]
            y_test = y[testind,:]
         
            # Normalize the input and out data based on the training set
            if scaler_x_minmax:
                scaler_x = MinMaxScaler()
                X_train = scaler_x.fit_transform(X_train)
                X_test = scaler_x.fit_transform(X_test) ## data leakage alert! test purpose only 
            elif scaler_x_robust:
                scaler_x = RobustScaler()
                X_train = scaler_x.fit_transform(X_train)
                X_test = scaler_x.transform(X_test)
            else:
                scaler_x = StandardScaler()
                X_train = scaler_x.fit_transform(X_train)
                X_test = scaler_x.transform(X_test)
            
            
            if use_scaler_y:
                scaler_y = StandardScaler()
            else:
                scaler_y = FunctionTransformer()
                
            y_train = scaler_y.fit_transform(y_train)
            y_test = scaler_y.transform(y_test)
            
        #######################################################################
        #                   Use PCA to reduce dimensionality                  #
        #######################################################################
            if use_PCA_x: 
                temp3 = np.empty((len(trainind),0))
                temp4 = np.empty((len(testind),0))
                logger.info('------------------------------------------------------------------------')
                logger.info('Computing principal components')
                n = CovariateNumIndCum
                for i in np.arange(1,len(n)):
                    temp = X_train[:, n[i-1]:n[i]]
                    if PCA_num!=0:
                        pca = PCA(n_components=PCA_num)
                        temp = pca.fit_transform(temp)
                        pca_explained_variance = np.cumsum(pca.explained_variance_ratio_)*100
                        logger.info(f'Adopting the first {PCA_num} princple components of input {i}')
                        formatted_variance = ", ".join(f"{var:.2f}" for var in pca_explained_variance)
                        logger.info('Cumulative explained variance:')
                        logger.info(f'{formatted_variance}')
                        logger.info(f"Dimensionality reduction: {CovariateNumInd[i-1]} --> {temp.shape[1]}")
                    elif PCA_variability_factor!=0:
                        pca = PCA(n_components=PCA_variability_factor)
                        temp = pca.fit_transform(temp)
                        
                        logger.info(f'Adopting components that explain {PCA_variability_factor*100}$\%$ of the varibility of input {i}')
                        logger.info(f"Dimensionality reduction: {CovariateNumInd[i-1]} --> {temp.shape[1]}")
                        
                    temp2 = pca.transform(X_test[:, n[i-1]:n[i]])
                    temp3 = np.concatenate((temp3,temp),axis = 1)
                    temp4 = np.concatenate((temp4,temp2),axis = 1)

                    
                    if flat_names_list[i-1]!='zws_xavg':
                        # plot the spatial pattern of the first principal component
                        eof_patterns = np.full((temp.shape[1], Nlats * Nlons), np.nan)  # Initialize with NaN
                        eof_patterns[:, land_mask.flatten()] = pca.components_  # Only fill valid locations
                        eof_patterns = eof_patterns.reshape((temp.shape[1], Nlats, Nlons))
                        
                        PCA_num_plot = min(temp.shape[1],6)
                        max_val = np.nanmax(np.abs(eof_patterns[0:PCA_num_plot]))
                        # Create a symmetric colormap around zero
                        norm = Normalize(vmin=-max_val, vmax=max_val)
                        if MOC_id ==2:
                            fig, axes = plt.subplots(nrows=round(PCA_num_plot), ncols=1, figsize=(18, 15))
                        elif MOC_id ==1:
                            fig, axes = plt.subplots(nrows=round(PCA_num_plot/3), ncols=3, figsize=(18, 12))
                        elif MOC_id ==3:
                            fig, axes = plt.subplots(nrows=round(PCA_num_plot), ncols=1, figsize=(18, 18))
                        
                        for ii, ax in enumerate(axes.flat):
                            if ii < PCA_num_plot:  # Ensure we only plot for the number of PCA components
                                im = ax.pcolormesh(lon_in,lat_in,eof_patterns[ii], cmap='RdBu_r', norm=norm, shading='auto')
                                explained_variance_pct = pca.explained_variance_ratio_[ii] * 100  # Convert to percentage
                                title = f"EOF {ii+1} ({explained_variance_pct:.2f}%)"
                                ax.set_title(title)
                                ax.set_xlabel('Longitude')
                                ax.set_ylabel('Latitude')
                                fig.colorbar(im, ax=ax, orientation='vertical')
                            else:
                                ax.axis('off')  # Turn off unused axes
                        plt.subplots_adjust(hspace=0.6)  # Adjust vertical spacing between rows
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, 'PCAs_' + flat_names_list[i-1]  + '.png'),dpi=300)
                        plt.show()
                        
                X_train = temp3
                X_test = temp4
                del temp, temp2, temp3, temp4
            
            if use_PCA_y: 
                logger.info('------------------------------------------------------------------------')
                logger.info('Computing principal components of MOC')
                if PCA_y_num!=0:
                    pca_y = PCA(n_components=PCA_y_num)
                    y_train = pca_y.fit_transform(y_train)
                    pca_explained_variance = np.cumsum(pca_y.explained_variance_ratio_)*100
                    logger.info(f'Adopting the first {PCA_y_num} princple components')
                    formatted_variance = ", ".join(f"{var:.2f}" for var in pca_explained_variance)
                    logger.info('Cumulative explained variance:')
                    logger.info(f'{formatted_variance}')
                    logger.info(f"Dimensionality reduction: {Psi.shape[1]} --> {y_train.shape[1]}")
                elif PCA_y_variability_factor!=0:
                    pca_y = PCA(n_components=PCA_y_variability_factor)
                    y_train = pca_y.fit_transform(y_train)
                    logger.info(f'Adopting components that explain {PCA_y_variability_factor*100}$\%$ of the varibility')
                    logger.info(f"Dimensionality reduction: {Psi.shape[1]} --> {y_train.shape[1]}")
                    
            
                y_test = pca_y.transform(y_test)
                scaler_y_allfolds.append(scaler_y)
                if Full_depth_MOC:
                    # plot the spatial pattern of the first principal components
                    eof_patterns = np.full((y_train.shape[1], Nlats_psi * Nlevs), np.nan)  # Initialize with NaN
                    eof_patterns[:, Psi_mask] = pca_y.components_  # Only fill valid locations
                    eof_patterns = eof_patterns.reshape((y_train.shape[1], Nlevs,Nlats_psi))

                    PCA_num_plot = min(y_train.shape[1],6)
                    max_val = np.nanmax(np.abs(eof_patterns[0:PCA_num_plot]))
                    # Create a symmetric colormap around zero
                    norm = Normalize(vmin=-max_val, vmax=max_val)
                    if MOC_id ==2:
                        fig, axes = plt.subplots(nrows=round(PCA_num_plot), ncols=1, figsize=(18, 18))
                    elif MOC_id ==1:
                        fig, axes = plt.subplots(nrows=round(PCA_num_plot), ncols=1, figsize=(18, 18))
                        ymax = 1037.19
                    elif MOC_id ==3:
                        fig, axes = plt.subplots(nrows=round(PCA_num_plot), ncols=1, figsize=(18, 18))
                        ymax = 1037.31
                    for i, ax in enumerate(axes.flat):
                        if i < PCA_num_plot:  # Ensure we only plot for the number of PCA components
                            im = ax.pcolormesh(lat_psi,rho2,eof_patterns[i], cmap='RdBu_r', norm=norm, shading='auto')
                            explained_variance_pct = pca_y.explained_variance_ratio_[i] * 100  # Convert to percentage
                            title = f"EOF {i+1} ({explained_variance_pct:.2f}%)"
                            ax.set_title(title)
                            ax.set_xlabel('Longitude')
                            ax.set_ylabel('Latitude')
                            ax.set_ylim(1035.25, ymax)
                            ax.invert_yaxis()
                            fig.colorbar(im, ax=ax, orientation='vertical')
                        else:
                            ax.axis('off')  # Turn off unused axes
                    plt.subplots_adjust(hspace=0.6)  # Adjust vertical spacing between rows
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'PCAs_MOC_fold'+str(fold_no)+'.png'),dpi=300)
                    plt.show()
                else:
                    # plot the spatial pattern of the first principal components
                    eof_patterns = pca_y.components_  # Initialize with NaN
                    PCA_num_plot = min(y_train.shape[1],6)
                    max_val = np.nanmax(np.abs(eof_patterns[0:PCA_num_plot]))
                    # Create a symmetric colormap around zero                   
                    fig, axes = plt.subplots(nrows=round(PCA_num_plot), ncols=1, figsize=(18, 18))
                    for i, ax in enumerate(axes.flat):
                        if i < PCA_num_plot:  # Ensure we only plot for the number of PCA components
                            im = ax.plot(lat_psi,eof_patterns[i])
                            explained_variance_pct = pca_y.explained_variance_ratio_[i] * 100  # Convert to percentage
                            title = f"EOF {i+1} ({explained_variance_pct:.2f}%)"
                            ax.set_title(title)
                            ax.set_xlabel('Latitude')
                            ax.set_ylabel('EOF magnitude')
                        else:
                            ax.axis('off')  # Turn off unused axes
                    plt.subplots_adjust(hspace=0.6)  # Adjust vertical spacing between rows
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'PCAs_MOC_fold'+str(fold_no)+'.png'),dpi=300)
                    plt.show()             

        #######################################################################
        #                                PCA ENDs                             #
        #######################################################################
        
        
            # Prepare the input branches
            if use_input_branches:
                n = CovariateNumIndCum
                X_train = [X_train[:, n[i-1]:n[i]] for i in range(1,len(n))]
                X_test = [X_test[:, n[i-1]:n[i]] for i in range(1,len(n))]
        
            # Generate a print
            logger.info('------------------------------------------------------------------------')
            # We use ensemble training for each fold of the cross-validation
            for ens_no  in np.arange(1,NNrepeats+1):
                logger.info(f'Training for fold {fold_no} ensemble {ens_no}...')
                skill, history, pred, R2_AllLats = train_model(X_train, y_train, X_test, y_test, scaler_y)
                trained_models.append((skill, skill, history.history, pred, R2_AllLats))
                if use_PCA_y:
                    y_pred_reconstructed = pca_y.inverse_transform(pred)
                    y_pred_reconstructed_allfolds.append(y_pred_reconstructed)
            if save_in_and_out:
                variables_dict = {'X_train': X_train, 'y_train': y_train, 'X_test':X_test, 'y_test':y_test}
                # Save variables to a .mat file
                sio.savemat(os.path.join(output_dir,'model_in_and_out'+str(fold_no) + '.mat'), variables_dict)
                
            # Increase fold number
            fold_no = fold_no + 1
            gpu_memory()
            tf.keras.backend.clear_session()
            gpu_memory()
            gc.collect()
            gpu_memory()
                
                
        logger.info('------------------------------------------------------------------------')
        logger.info(f'Training with {num_folds}-fold cross-validation finished!') 
        logger.info('------------------------------------------------------------------------')
        
        
        
        #%% Check if the model skill varies too much among different emsembles and folds
        skills = [trained_model[1] for trained_model in trained_models[0:num_folds*NNrepeats]]
        median_skill = np.median(skills)
        median_index = skills.index(median_skill)
        minimum_skill = np.min(skills)
        min_index = skills.index(minimum_skill)
        std_dev_skill = np.std(skills)
        
        formatted_skills = [f"{skill:.4f}" for skill in skills]
        logger.info(f"Losses in testing set: {formatted_skills}")
        fold_median = median_index // NNrepeats + 1  # Compute fold number
        ensemble_median = median_index % NNrepeats + 1  # Compute ensemble number
        logger.info(f"Median Loss is: {median_skill:.4f}, which occurs at Fold {fold_median} Ensemble {ensemble_median}")
        fold_min = min_index // NNrepeats + 1  # Compute fold number
        ensemble_min = min_index % NNrepeats + 1  # Compute ensemble number
        logger.info(f"Minimum Loss is: {minimum_skill:.4f}, which occurs at Fold {fold_min} Ensemble {ensemble_min}")
        logger.info(f"Standard Deviation of the Loss is: {std_dev_skill:.4f}")
        ratio = std_dev_skill/median_skill * 100
        logger.info(f"Standard Deviation/Median: {ratio:.2f}%")
        
        # Check if the standard deviation to median skill ratio is higher than 30%
        if ratio > 30:
            warning_message = ("Warning: The standard deviation of model skill across "
                               "folds and ensembles is large!\nStandard Deviation/Median Skill: {ratio:.2f}%").format(ratio=ratio)
            with open(os.path.join(output_dir,'Warning_'+covariate_names_str+'.txt'), "w") as file:
                file.write(warning_message)
            with open(os.path.join(output_dir,'..','Warning_'+covariate_names_str+'.txt'), "w") as file:
                file.write(warning_message)

        
        # Plot the loss function during training for all folds
        # First, find the global minimum and maximum loss values across all folds
        min_loss = min(min(trained_models[i][2]['loss']+trained_models[i][2]['val_loss']) for i in range(num_folds))
        max_loss = max(max(trained_models[i][2]['loss']+trained_models[i][2]['val_loss']) for i in range(num_folds))
        
        plt.figure(figsize=(9, 12))
        
        # Plot the first fold outside the loop to avoid repeating the legend setting
        plt.subplot(num_folds, 1, 1)
        for nn in range(NNrepeats):
            plt.plot(trained_models[nn][2]['loss'],'-k')
            plt.plot(trained_models[nn][2]['val_loss'],'-r')
        plt.ylabel('Loss')
        plt.yscale('log')  # Set logarithmic scale
        plt.ylim(min_loss, max_loss)  # Set the same y-limits for all subplots
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Now plot the remaining folds
        for n in range(1, num_folds):
            plt.subplot(num_folds, 1, n+1)
            for nn in range(NNrepeats):
                plt.plot(trained_models[n*NNrepeats+nn][2]['loss'],'-k')
                plt.plot(trained_models[n*NNrepeats+nn][2]['val_loss'],'-r')
            plt.ylabel('Loss')
            plt.yscale('log')  # Set logarithmic scale
            plt.ylim(min_loss, max_loss)  # Set the same y-limits for all subplots
        
        plt.xlabel('Epochs')
        plt.subplots_adjust(hspace=0.5)  # Adjust space between plots if needed
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'TrainingLoss_' + covariate_names_str  + '.png'),dpi=300)
        plt.show()
        
        #%% plot the true MOC, predicted MOC, and the prediction error
        
        # The prediction is made by combining the testing sets of all folds and averaging across ensembles
        if use_PCA_y:
            Model_preds = [np.concatenate([scaler_y_allfolds[n].inverse_transform(y_pred_reconstructed_allfolds[n*NNrepeats + i]) 
                                           for n in range(num_folds)]) for i in range(NNrepeats)]
        else:
            Model_preds = [np.concatenate([trained_models[n*NNrepeats + i][3] for n in range(num_folds)]) for i in range(NNrepeats)]
            
        Model_pred = np.mean(Model_preds,axis=0)
        Model_error = Model_pred - y
        
        if not use_local_inputs and not Full_depth_MOC:
            
            X_plot,Y_plot = np.meshgrid(lat_psi,np.arange(1,Nsamps+1)/12)
            ft = 15
            if LPF_month > 0:
               vmin = -6
               vmax = 6 
            else:
               vmin = -12
               vmax = 12
            foldlen = Nsamps/num_folds/12
            
            
            
            def create_and_save_plot(y_limits, filename):
                fig, axs = plt.subplots(1, 3, figsize=(20, 12), num='1')
                
                # Plot 1: MOC strength
                c = axs[0].pcolormesh(X_plot, Y_plot, y, shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
                fig.colorbar(c, ax=axs[0], shrink=0.7)
                axs[0].plot([lat_psi[0],lat_psi[-1]],[foldlen,foldlen],'--k')
                axs[0].plot([lat_psi[0],lat_psi[-1]],[foldlen*2,foldlen*2],'--k')
                axs[0].plot([lat_psi[0],lat_psi[-1]],[foldlen*3,foldlen*3],'--k')
                axs[0].plot([lat_psi[0],lat_psi[-1]],[foldlen*4,foldlen*4],'--k')
                axs[0].set_xlabel('Latitude', fontsize=ft)
                axs[0].set_ylabel('Time (years)', fontsize=ft)
                axs[0].set_title('MOC strength', fontsize=ft)
                axs[0].set_ylim(y_limits)
            
                # Plot 2: Prediction
                c = axs[1].pcolormesh(X_plot, Y_plot, Model_pred, shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
                fig.colorbar(c, ax=axs[1], shrink=0.7)
                axs[0].plot([lat_psi[0],lat_psi[-1]],[foldlen,foldlen],'--k')
                axs[0].plot([lat_psi[0],lat_psi[-1]],[foldlen*2,foldlen*2],'--k')
                axs[0].plot([lat_psi[0],lat_psi[-1]],[foldlen*3,foldlen*3],'--k')
                axs[0].plot([lat_psi[0],lat_psi[-1]],[foldlen*4,foldlen*4],'--k')
                axs[1].set_xlabel('Latitude', fontsize=ft)
                axs[1].set_title('Prediction', fontsize=ft)
                axs[1].set_ylim(y_limits)
            
                # Plot 3: Error map
                # c = axs[2].pcolormesh(X_plot, Y_plot, Model_error, shading='auto', cmap='RdBu_r', vmin=vmin*0.5, vmax=vmax*0.5)
                c = axs[2].pcolormesh(X_plot, Y_plot, Model_error, shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
                fig.colorbar(c, ax=axs[2], shrink=0.7)
                axs[0].plot([lat_psi[0],lat_psi[-1]],[foldlen,foldlen],'--k')
                axs[0].plot([lat_psi[0],lat_psi[-1]],[foldlen*2,foldlen*2],'--k')
                axs[0].plot([lat_psi[0],lat_psi[-1]],[foldlen*3,foldlen*3],'--k')
                axs[0].plot([lat_psi[0],lat_psi[-1]],[foldlen*4,foldlen*4],'--k')
                axs[2].set_xlabel('Latitude', fontsize=ft)
                axs[2].set_title('Error map', fontsize=ft)
                axs[2].set_ylim(y_limits)
            
                # Save the plot
                plt.savefig(os.path.join(output_dir, filename), dpi=300)
                plt.show()
                plt.close(fig)
            
            # Original y-axis limits plot
            create_and_save_plot(None, 'Pred_YTMap_' + covariate_names_str + '.png')
            
            if LPF_month<=60:
                # First 100 years plot
                create_and_save_plot((0, 100), 'Pred_YTMap_First100Years_' + covariate_names_str + '.png')
                
                # Last 200 years plot
                create_and_save_plot((Nsamps/12 - 200, Nsamps/12), 'Pred_YTMap_Last200Years_' + covariate_names_str + '.png')
            
            if LPF_month==0:
                # First 10 years plot
                create_and_save_plot((0, 10), 'Pred_YTMap_First10Years_' + covariate_names_str + '.png')
        
        
        #%% Calculate skill metrics and plot R2 at different latitudes
        
        r2_lats_max = []
        r2_lats_min = []
        R2_each_fold_ens = [trained_models[i][4] for i in range(0,num_folds*NNrepeats)]
        out_num = [np.array(R2_each_fold_ens[i]).shape for i in range(0,num_folds*NNrepeats)]
        out_num_min = np.array(out_num).min()
        for latind in range(out_num_min):
            temp = [R2_each_fold_ens[i][latind] for i in range(0,num_folds*NNrepeats)]
            r2_lats_max.append(np.max(temp))
            r2_lats_min.append(np.min(temp))

            
        r2 = r2_score(y, Model_pred)
        corr_coef, _ = pearsonr(y.flatten(), Model_pred.flatten())
        rmse = np.sqrt(mean_squared_error(y, Model_pred))
        logger.info(f"R-squared (Testing): {r2:.3f}")
        logger.info(f"Correlation coefficient (Testing): {corr_coef:.3f}")
        logger.info(f"RMSE (Testing): {rmse:.3f}")
        
        # Calculate R-squared value, correlation coefficient, and RMSE at each latitude
        r2_lats = []
        corr_coef_lats = []
        rmse_lats = []
        for latind in range(y.shape[1]):
            y_lat = y[:, latind];
            y_pred_lat = Model_pred[:, latind];
            r2_lats.append(r2_score(y_lat, y_pred_lat))
            corr_coef_lats.append(pearsonr(y_lat.flatten(), y_pred_lat.flatten()))
            rmse_lats.append(np.sqrt(mean_squared_error(y_lat, y_pred_lat)))
             
        if Full_depth_MOC:
            r2_yz = np.full((Nlats_psi * Nlevs), np.nan)  # Initialize with NaN
            r2_yz[Psi_mask] = r2_lats # Only fill valid locations
            r2_yz = r2_yz.reshape((Nlevs,Nlats_psi))
            if MOC_id ==2:
                plt.figure(figsize=(10, 6))
            elif MOC_id ==1:
                plt.figure(figsize=(24, 6))
            elif MOC_id ==3:
                plt.figure(figsize=(18, 6))
            plt.pcolormesh(lat_psi,rho2,r2_yz, cmap='RdYlBu_r', shading='auto', vmin=0, vmax=1)
            plt.gca().invert_yaxis()
            title = ('$R^2$')
            plt.title(title)
            plt.xlabel('Latitude')
            plt.ylabel('Potential density')
            plt.colorbar(orientation='vertical')
            plt.clim()
            plt.savefig(os.path.join(output_dir, 'Metrics_R2_YZ_' + covariate_names_str  + '.png'),dpi=300)
            plt.savefig(os.path.join(output_dir, '..','Metrics_R2_YZ_' + covariate_names_str  + '.png'),dpi=300)
            plt.show()
                
        if not use_local_inputs and not Full_depth_MOC:
            # plot the skill metrics at each latitude
            plt.figure(figsize=(6, 4))
            plt.plot(lat_psi, r2_lats, 'k', label='Including all folds and ensembles')  # Plot the main line
            if not use_PCA_y:
                plt.fill_between(lat_psi, r2_lats_min, r2_lats_max, color='gray', alpha=0.5, label='Range among all folds and ensembles')  # Shade between max and min 
            plt.ylim(0, 1)
            plt.xlabel('Latitude')
            plt.ylabel('$R^2$')
            plt.title(MOC_str  +LPF_str+'; '+covariate_names_str)
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(output_dir, 'Metrics_R2atAllLats_' + covariate_names_str  + '.png'),dpi=300)
            plt.savefig(os.path.join(output_dir, '..','Metrics_R2atAllLats_' + covariate_names_str  + '.png'),dpi=300)
            plt.show()
            
            # plot the skill metrics at each latitude
            plt.figure(figsize=(6, 4))
            plt.plot(lat_psi, np.array(corr_coef_lats)[:,0], 'k')  # Plot the main line
            plt.ylim(0, 1)
            plt.xlabel('Latitude')
            plt.ylabel('Correlation coefficient')
            plt.title(MOC_str  +LPF_str+'; '+covariate_names_str)
            plt.grid()
            plt.savefig(os.path.join(output_dir, 'Metrics_cc_atAllLats_' + covariate_names_str  + '.png'),dpi=300)
            plt.savefig(os.path.join(output_dir, '..','Metrics_cc_atAllLats_' + covariate_names_str  + '.png'),dpi=300)
            plt.show()
            
            
            # plot the skill metrics at each latitude
            plt.figure(figsize=(6, 4))
            plt.plot(lat_psi, rmse_lats, 'k')  # Plot the main line
            plt.xlabel('Latitude')
            plt.ylabel('Root mean squared error (Sv)')
            plt.title(MOC_str  +LPF_str+'; '+covariate_names_str)
            plt.grid()
            plt.savefig(os.path.join(output_dir, 'Metrics_rmse_atAllLats_' + covariate_names_str  + '.png'),dpi=300)
            plt.savefig(os.path.join(output_dir, '..','Metrics_rmse_atAllLats_' + covariate_names_str  + '.png'),dpi=300)
            plt.show()
        
        
        
        variables_dict = {'R2_each_fold_ens':R2_each_fold_ens,
                          'r2_test': r2, 'corr_coef_test': corr_coef, 'rmse_test': rmse,
                          'r2_lats_max':r2_lats_max, 'r2_lats_min':r2_lats_min,
                          'r2_test_lats': r2_lats, 'corr_coef_test_lats': corr_coef_lats, 'rmse_test_lats': rmse_lats}
        
        # Save variables to a .mat file
        sio.savemat(os.path.join(output_dir, 'NNmetrics_' + covariate_names_str +  '.mat'), variables_dict)
        
        variables_dict = {'y': y, 'y_pred': Model_pred, 'y_pred_all_ensembles':Model_preds}
        
        # Save variables to a .mat file
        sio.savemat(os.path.join(output_dir, 'NNpredictions_' + covariate_names_str +  '.mat'), variables_dict)
        
        if Full_depth_MOC:
            sio.savemat(os.path.join(output_dir, 'Psi_mask.mat'), {'Psi_mask': Psi_mask})
        #%% plots for explainable AI
        if use_LRP:
            os.makedirs(os.path.join(output_dir,'LRP_folds_avg'), exist_ok=True)
            os.makedirs(os.path.join(output_dir,'LRP_abs_folds_avg'), exist_ok=True)
            
            temp3 = []
            temp6 = []
            temp7 = []
            temp8 = []
            temp9 = np.empty((y_test.shape[1],0))
            for fold in np.arange(1,6):
                temp = sio.loadmat(os.path.join(output_dir,'LRP_fold'+str(fold),'LatwideRelevance.mat'))
                
                temp2 = temp['mean_absolute_relevances']
                temp3.append(temp2)
                del temp2
                
                temp2 = temp['mean_relevances']
                temp6.append(temp2)
                del temp2
                
                temp2 = temp['relevances_std']
                temp7.append(temp2)
                del temp2
                
                temp2 = temp['relevances_abs_std']
                temp8.append(temp2)
                del temp2
                
                temp2 = temp['relevances_sum']
                
                temp9 = np.concatenate((temp9,temp2),axis = 1)
                
                del temp2
                
                intv = np.squeeze(temp['intv'])
                del temp
                
            mean_absolute_relevances = np.mean(np.array(temp3),axis=0)
            mean_relevances = np.mean(np.array(temp6),axis=0)
            relevances_std = np.mean(np.array(temp7),axis=0)
            relevances_abs_std = np.mean(np.array(temp8),axis=0)
            relevances_sum = temp9
            
            sio.savemat(os.path.join(output_dir,'LRP_folds_avg','LatwideRelevance.mat'), 
                        {'mean_relevances': mean_relevances, 'mean_absolute_relevances': mean_absolute_relevances,
                         'relevances_std':relevances_std,'relevances_abs_std':relevances_abs_std,'relevances_sum':relevances_sum,
                         'land_mask': land_mask, 'Nlats':Nlats, 'Nlons':Nlons,
                         'OutFeatureNum':out_num, 'intv':intv,
                        'CovariateNumIndCum':CovariateNumIndCum,'covariate_names':covariate_names})
                   
                    
            ind = 0
            for name in covariate_names.split(','):
                ind = ind + 1
                n = CovariateNumIndCum
                temp = mean_relevances[:,n[ind-1]:n[ind]]
                temp2 = mean_absolute_relevances[:,n[ind-1]:n[ind]]
                if name!= 'zws_xavg':        
                    mean_relevances_yz = np.full((y_train.shape[1],Nlats*Nlons), np.nan)  # Initialize with NaN
                    mean_relevances_yz[:,land_mask] = temp  # Only fill valid locations
                    mean_relevances_yz = mean_relevances_yz.reshape((y_test.shape[1],Nlats, Nlons))
                    
                    mean_absolute_relevances_yz = np.full((y_train.shape[1],Nlats*Nlons), np.nan)  # Initialize with NaN
                    mean_absolute_relevances_yz[:,land_mask] = temp2  # Only fill valid locations
                    mean_absolute_relevances_yz = mean_absolute_relevances_yz.reshape((y_test.shape[1],Nlats, Nlons))
                    
                    
                    # plot the spatial pattern of relevance score
                    for i in range(0,y_test.shape[1],intv):
                        if MOC_id ==2:
                            plt.figure(figsize=(15, 3))
                            vlim = 0.0006
                        elif MOC_id ==1:
                            plt.figure(figsize=(10, 8))
                            vlim = 0.0006
                        elif MOC_id ==3:
                            plt.figure(figsize=(15, 6))
                            vlim = 0.0003
                            
                        plt.title(f"{name}; Mean Relevance Score for MOC at latitude {lat_psi[i]:.1f}; Averaged among folds")
                        plt.pcolormesh(lon_in,lat_in,mean_relevances_yz[i], cmap='RdBu_r',vmin=-vlim, vmax=vlim)  # Adjust reshape as necessary
                        plt.colorbar()
                        plt.plot([lon_in[0],lon_in[-1]],[lat_psi[i],lat_psi[i]],'--k')
                        plt.xlabel('Longitude')
                        plt.ylabel('Latitude')
                        plt.savefig(os.path.join(output_dir, 'LRP_folds_avg', name  +'_LatwideRelevance_' + str(i) + '.png'),dpi=300)
                        plt.show()
                        
                        
                        
                    for i in range(0,y_test.shape[1],intv):
                        # plot the spatial pattern of absolute relevance score
                        if MOC_id ==2:
                            plt.figure(figsize=(15, 3))
                            vlim = 0.004
                        elif MOC_id ==1:
                            plt.figure(figsize=(10, 8))
                            vlim = 0.004
                        elif MOC_id ==3:
                            plt.figure(figsize=(15, 6))
                            vlim = 0.002
                            
                            
                        plt.title(f"{name}; Mean Absolute Relevance Score for MOC at latitude {lat_psi[i]:.1f}; Averaged among folds")
                        plt.pcolormesh(lon_in,lat_in,mean_absolute_relevances_yz[i], cmap='RdYlBu_r',vmin=0, vmax=vlim)  # Adjust reshape as necessary
                        plt.colorbar()
                        plt.plot([lon_in[0],lon_in[-1]],[lat_psi[i],lat_psi[i]],'--k')
                        plt.xlabel('Longitude')
                        plt.ylabel('Latitude')
                        plt.savefig(os.path.join(output_dir, 'LRP_abs_folds_avg', name  +'_LatwideRelevance_' + str(i) + '.png'),dpi=300)
                        plt.show()
                        
                        
                        
                        
                        
                else:
                    for i in range(0,y_test.shape[1],intv):
                        
                        # plot relevance score of zonal averaged quantity
                        if MOC_id ==2:
                            plt.figure(figsize=(10, 8))
                            vlim = 0.0008
                        elif MOC_id ==1:
                            plt.figure(figsize=(12, 6))
                            vlim = 0.0004
                        elif MOC_id ==3:
                            plt.figure(figsize=(10, 8))
                            vlim = 0.0004
                        plt.title(f"{name}; latitude {lat_psi[i]:.1f}; Averaged among folds")
                        plt.plot(lat_in,temp[i,:])  
                        plt.xlabel('Latitude')
                        plt.ylabel('Mean Relevance Score')
                        plt.ylim([-vlim ,vlim])
                        y1, y2 = plt.ylim()
                        plt.plot([lat_psi[i],lat_psi[i]],[y1, y2],'--k')
                        plt.savefig(os.path.join(output_dir, 'LRP_folds_avg', name  +'_LatwideRelevance_' + str(i) + '.png'),dpi=300)
                        plt.show()
                        
                    for i in range(0,y_test.shape[1],intv):
                        # plot absolute relevance score of zonal averaged quantity
                        if MOC_id ==2:
                            plt.figure(figsize=(10, 8))
                            vlim = 0.008
                        elif MOC_id ==1:
                            plt.figure(figsize=(12, 6))
                            vlim = 0.004
                        elif MOC_id ==3:
                            plt.figure(figsize=(10, 8))
                            vlim = 0.004
                        plt.title(f"{name}; latitude {lat_psi[i]:.1f}; Averaged among folds")
                        plt.plot(lat_in,temp2[i,:])  
                        plt.xlabel('Latitude')
                        plt.ylabel('Mean Absolute Relevance Score')
                        plt.ylim([0, vlim])
                        y1, y2 = plt.ylim()
                        plt.plot([lat_psi[i],lat_psi[i]],[y1, y2],'--k')
                        plt.savefig(os.path.join(output_dir, 'LRP_abs_folds_avg', name  +'_LatwideRelevance_' + str(i) + '.png'),dpi=300)
                        plt.show()

                
                
                
        #%% copy this script to the directory that stores the trained NN
        import shutil
        
        def copy_script(target_directory):
            # Get the current script file path
            # current_script = __file__
            current_script = script_path_and_name
            
            # Ensure the target directory exists, create if it does not
            os.makedirs(target_directory, exist_ok=True)
            
            # Define the target path for the script
            target_path = os.path.join(target_directory, os.path.basename(current_script))
            
            # Copy the script
            shutil.copy(current_script, target_path)
            logger.info(f"Script copied to {target_path}")
        
        # Example usage
        copy_script(output_dir)
            
        
        # Copy the logfile
        shutil.copy(os.path.join(analysis_folder, logger_name), output_dir)
        
    if file_handler:
        logger.removeHandler(file_handler)
        logger.removeHandler(console_handler)
        file_handler.close()  
    if os.path.exists(os.path.join(analysis_folder, logger_name)):
        os.remove(os.path.join(analysis_folder, logger_name))


#%% run this script with default settings
MOC_id = 1
covariate_names = "obp,ssh,zws"
neural_network_training(MOC_id,covariate_names)
