from sdv.single_table import CTGANSynthesizer, CopulaGANSynthesizer

def classical_ctgan(original_data, epochs, metadata, generator_lr, discriminator_lr, batch_size, input_size,
                    num_hidden_layers_gen, num_dimensions_gen, num_hidden_layers_disc, num_dimensions_disc):

   # metadata = get_metadata(original_data)
    generator_dim = tuple([num_dimensions_gen for i in range(num_hidden_layers_gen)])
    discriminator_dim = tuple([num_dimensions_disc for i in range(num_hidden_layers_disc)])

    ctgan_synthesizer = CTGANSynthesizer(
        metadata, # required
        enforce_rounding=True,
        epochs=epochs,
        verbose=True,
        generator_lr=generator_lr,
        discriminator_lr=discriminator_lr,
        batch_size=batch_size,
        generator_dim= generator_dim,
        discriminator_dim= discriminator_dim,
        embedding_dim= input_size,
        pac = 1
    )
    ctgan_synthesizer.fit(original_data)
    num_parameters = ctgan_synthesizer.get_parameters()
    return ctgan_synthesizer.sample(num_rows=len(original_data)),num_parameters


def classical_copula_gan(original_data, epochs, metadata, generator_lr, discriminator_lr, batch_size, input_size,
                         num_hidden_layers_gen, num_dimensions_gen, num_hidden_layers_disc, num_dimensions_disc):

    #metadata = get_metadata(original_data)
    generator_dim = tuple([num_dimensions_gen for i in range(num_hidden_layers_gen)])
    discriminator_dim = tuple([num_dimensions_disc for i in range(num_hidden_layers_disc)])

    copulagan_synthesizer = CopulaGANSynthesizer(
        metadata, # required
        enforce_rounding=True,
        epochs=epochs,
        verbose=True,
        generator_lr=generator_lr,
        discriminator_lr=discriminator_lr,
        batch_size=batch_size,
        generator_dim= generator_dim,
        discriminator_dim= discriminator_dim,
        embedding_dim=input_size,
        pac=1
    )
    copulagan_synthesizer.fit(original_data)
    num_parameters = copulagan_synthesizer.get_parameters()
    return copulagan_synthesizer.sample(num_rows=len(original_data)), num_parameters