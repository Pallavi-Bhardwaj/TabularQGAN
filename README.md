# TabularQGAN
TabularQGAN for submission

The subdirectory vae-bgm_data_generator contains the code used to generate the VAE-BGM classical benchmarks. This code is a modified version of the code found here:
[(https://github.com/Patricia-A-Apellaniz/vae-bgm_data_generator/tree/main]) 

The modifications were:
* adding a additional data handlers for the mimic and adult census data sets in  'ctgan_datasets.py' 
* Adding functions to 'utils.py' to create different hyperparamter configurations 
* Adapting 'main_generator.py to accept the changes main in the other parts of the code
