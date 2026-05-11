
import json
import os
import sys
import time
import hashlib
import math 
from qugen.main.data.helper import tabular_feature_names, run_external_classical_tabular
import pandas as pd
import numpy as np

from sdv.metadata import SingleTableMetadata
from qugen.main.generator.tabular_qgan_model_handler import (
    TabularQGANModelHandler,
)
from qugen.main.data.data_handler import load_data
from qugen.main.data.data_handler import TabularDataTransformer as tdf


script                = sys.argv[0]
data_set_name         = sys.argv[1]
model_type            = sys.argv[2]
circuit_depth         = int(sys.argv[3])
n_epochs              = int(sys.argv[4])
lr_generator          = float(sys.argv[5]) 
lr_discriminator      = float(sys.argv[6]) 
batch_size_fraction   = float(sys.argv[7])
classical_model_size  = 'small' if len(sys.argv)<9 else sys.argv[8]
random_seed           = int(sys.argv[9])
# root_path = os.getcwd()
root_path = os.getcwd() + '/apps/ehr'
data_set_path = f"{root_path}/training_data/{data_set_name}.csv"
meta_path = f"{root_path}/training_data/{data_set_name}_meta.json"


#for quantum model what is the metric which picks the best epoch  
decision_metric = 'overall_metric'

#load the meta data for the training data 
with open(meta_path, 'r') as f:
        meta = json.load(f)
data_spec = meta['data_spec']
n_qubits = meta['n_qubits']
column_names = meta['column_names']
csv_column_name = 'combined'
data = pd.read_csv(data_set_path, dtype={'combined': str})['combined'].values
data = np.array(list(list(map(int, list(d))) for d in data))
data_lenght = len(data)
batch_size = math.ceil((data_lenght*batch_size_fraction) / 2.) * 2 

#transforming the training data for plotting. 
human_readable_data  = tdf.decode_tabular_data(pd.DataFrame(data[:data_lenght]), data_spec, column_names)

#minor data cleaning for ensuring numeric columns are floats. 
for numeric_col in ['AGE', 'age', 'INTAKE TIME', 'intake time', 'hours.per.week', ]:
    if numeric_col in human_readable_data.columns:
        human_readable_data[numeric_col] = human_readable_data.loc[:, numeric_col].astype('float')


# if training the TabularQGAN model 
if model_type == 'qgan':
    model = TabularQGANModelHandler()
    model.build(
        model_name="tabular",
        data_set_name=data_set_name,
        n_qubits=  n_qubits,
        circuit_depth=circuit_depth,
        transformation="tabular",
        circuit_type="tabular",
        data_spec=data_spec,
        n_registers=None,
        random_seed=random_seed,
        column_names=column_names
    )

    model.train(
        data,
        n_epochs=n_epochs,
        initial_learning_rate_generator = lr_generator,
        initial_learning_rate_discriminator =lr_discriminator,
        batch_size=batch_size
    )
    #only using best samples for evaluation
    model_name = model.model_name
    evaluation_dict = model.evaluate_tabular(data, num_samples=data_lenght)
    evaluation_samples, best_epoch, best_metric = evaluation_dict[decision_metric]
    file_path = "experiments/"+ model_name + "/"

    np.save(f'{file_path}best_samples.npy', evaluation_samples)
    # transform data back to reable and save samples from best model 
    synthetic_data = tdf.decode_tabular_data(pd.DataFrame(evaluation_samples), data_spec, column_names)
    synthetic_data.to_csv(f'{file_path}synthetic_data_{data_set_name}_{model_name}.csv')
    


feature_names, input_size, cat_features, numerical_features, numerical_qubits = tabular_feature_names(column_names, data_spec)
 #training classical models 
if model_type != 'qgan':
    if classical_model_size == 'small':
        input_size = int(input_size)
        num_hidden_layers_gen=int(circuit_depth)
        num_dimensions_gen= 2 * input_size
        num_hidden_layers_disc= 2
        num_dimensions_disc= 2 * input_size
    else: 
        # hardcoding the external gan parameters to match prev experiments 
        if model_type == 'icgan':
            input_size = int(input_size)
        else:
            input_size = 128
        num_hidden_layers_gen= circuit_depth
        num_dimensions_gen= 256 
        num_hidden_layers_disc= 2
        num_dimensions_disc= 256  
    time_str = str(time.time()).encode('utf-8')
    uniq = hashlib.md5(time_str).hexdigest()[:4]
    model_name = f'{model_type}_{data_set_name}_{uniq}'
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(human_readable_data)
    file_path = "experiments_classical/" + model_name + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    plots_path = f"{os.getcwd()}/{file_path}"

    if model_type == 'ctgan' or model_type == 'copulagan':
        synthetic_data, col_shapes, col_pair, overall_score = run_external_classical_tabular(human_readable_data,
                                                                                            metadata,
                                                                                            model_type,
                                                                                            n_epochs,
                                                                                            lr_generator,
                                                                                            lr_discriminator,
                                                                                            batch_size,
                                                                                            input_size,
                                                                                            num_hidden_layers_gen= num_hidden_layers_gen,
                                                                                            num_dimensions_gen= num_dimensions_gen,
                                                                                            num_hidden_layers_disc= num_hidden_layers_disc,
                                                                                            num_dimensions_disc= num_dimensions_disc)
        df_overall = pd.DataFrame([[n_epochs, overall_score]], columns=['Epoch','Overall_metric'])
        

   


if model_type == 'ctgan' or model_type == 'copulagan' :
    synthetic_data.to_csv(f'{plots_path}synthetic_data_{data_set_name}_{model_name}.csv')
    df_overall.to_csv(f'{plots_path}overall_metric_{data_set_name}_{model_name}.csv')
if model_type != "qgan":
    keys = ['script',
            'data_set_name',
            'model_type',
            'circuit_depth',
            'n_epochs',
            'lr_generator',
            'lr_discriminator',
            'batch_size_fraction',
            'model_size']
    meta_dict = dict(zip(keys, sys.argv))

    with open(f'{plots_path}meta.json', 'w') as fp:
        json.dump(meta_dict, fp)



     



