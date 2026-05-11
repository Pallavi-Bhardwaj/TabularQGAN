
import pandas as pd

from qugen.main.generator.classical_gan_model_handler import ClassicalGAN
from qugen.main.data.data_handler import TabularDataTransformer


def run_classical_tabular(original_data, data_spec, epochs, generator_lr, discriminator_lr, batch_size, numerical_columns,
                          categorical_columns, input_size, num_hidden_layers_gen, num_dimensions_gen,
                          num_hidden_layers_disc, num_dimensions_disc, model_path):

    classicalgan = ClassicalGAN(input_size,num_hidden_layers_gen, num_dimensions_gen,
                                num_hidden_layers_disc, num_dimensions_disc, model_path)
    tdf = TabularDataTransformer(data_spec)

    # transform numerical and categorical columns with MinMaxScaler and OneHotEncoder
    training_data, num_pipeline, cat_pipeline = tdf.transform_classical_gan_data(original_data, numerical_columns,
                                                                         categorical_columns)
    classical_gan_params, classical_disc_params =  classicalgan.train(batch_size, training_data,
                                                                                                epochs, generator_lr,
                                                                                                discriminator_lr)
    print(f"internal classical GAN generator params: {classical_gan_params}")
    print(f"internal classical GAN discriminator params: {classical_disc_params}")



    # vanillagan_samples = classicalgan.predict(len(original_data))
    # decoded_samples = tdf.inverse_transform_classical_gan_data(num_pipeline, cat_pipeline, vanillagan_samples,
    #                                                            numerical_columns, categorical_columns)
    # evaluate the icgan model
    
    eval_dict = classicalgan.evaluate_icgan(original_data, data_spec, epochs, numerical_columns, categorical_columns,
                                            num_pipeline, cat_pipeline, model_path)

    best_samples, best_epoch, best_metric = eval_dict["overall_metric"]
    pd.DataFrame(best_samples).to_csv( f"{model_path}/synthetic_data.csv")
    return best_samples, best_epoch, best_metric

