 ## Introduction
 This code is intended to be a companion to the submitted paper 'TabularQGAN: A Quantum Generative Model for Tabular Data.'
It provides the ability to train the TabularQGAN model and the classical benchmark models. 
 
It is an adaptation of the open source code, qugen. For full documentation on the qugen library please see https://qutacquantum.github.io/qugen/
 
 
 ## Installation 

1) Create a virtual environment, e.g. using ``conda create --name tabqgan_env python=3.9.12``. Python 3.9 or later is supported.
2) Activate the enviroment, e.g ``source activate tabqgan_env``.
3) Run ``pip install .`` or ``pip install -e .`` to install it in editable mode. Additionally, (yaml file with package dependencies for conda environment is also provided)


## Instructions for training and running models

The script to train the modeles is called 'apps/ehr/train_tabularqgan.py' which is run from the command line. 

The training data, model type, and hyperparameters can all be specified via command line arguments. An output of a sample of synthetic data from the trained model. 

The tabularQGAN model also saves as pickle files the parameters at each epoch of training and as a csv  and a plot the asscoiated overall metric and KL metric for each epoch of training. 

Each model type also has a model handler, they are found in qugen/main/generator and are imported by the training script as needed. 

Steps:
1.  Run the file 'apps/ehr/data_ingestion_tabular_adults_census.py' which takes the raw data and converts it to bitstrings.  For the script to run the processed training data must be saved in a directory 'apps/ehr/training_data'.
2. Select the model type, data set and hyperparameters via command line: 


  The command line arguments for the script are as follows: 
  |Number | Name | Description |
  |---|---|---|
  | 0. | script | hyperparameter_train_discrete.py |
  | 1. | data_set_name | str: Options are 'adult_census_10', 'adult_census_15', 'adults_census_10_non_boolean', 'adults_census_15_non_boolean' | 
  | 2. | model_type | str: Options are 'qgan', 'ctgan' and 'copulagan' | 
  | 3. | circuit_depth | int:  for number of layers |
  | 4. | n_epochs| int: for number of epochs to train each model | 
  | 5. | lr_generator | float:  learning rate for the generator | 
  | 6. | lr_discriminator | float:  learning rate for the discriminator | 
  | 7. | batch_size_fraction | float:  batch size fraction for the training | 
  | 8. | classical_model_size | str:  options 'small', 'large' , for the layer width for the classical model.  | 
  | 9. | random_seed | int: random seed for initialising jax | 


  ## Example 
  Here is an example way to run the script: 

  `python -u apps/ehr/train_tabularqgan.py adults_census_10_non_boolean qgan 2 3000 0.1 0.05  0.1 small 11` 
  
3. The synthetic data is found at 'experiments/[model_name]/synethetic_data_[model_name].csv' for the tabularQGAN model and 
'experiments_classical/[model_name]/synethetic_data_[model_name].csv' for the classical models. Additional files produced in the experiments folder are-
 pickle files for each iteration, loss curve for KL divergence and overall metric and csv files with best overall metric and kl divergence

## Notes

For convenience we have also included a json file with the best found configurations per model and dataset as reported in the paper, called 'best_config.json'.

Only the data processeing code for the adult census data set is provided here as the MIMIC-III dataset cannot not be distributed. See https://physionet.org/content/mimiciii/1.4/ for details.

## Contact

pallavi.bhardwaj@sap.com



