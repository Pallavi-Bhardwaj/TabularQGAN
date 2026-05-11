
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sdv.evaluation.single_table import evaluate_quality

from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset

from qugen.main.data.data_handler import TabularDataTransformer
from qugen.main.data.helper import kl_divergence_from_data_tabular, get_metadata
from qugen.main.generator.classical_generator_tabular import Classical_Generator as Generator
from qugen.main.discriminator.classical_discriminator import Classical_Discriminator as Discriminator


class ClassicalGAN():

    def __init__(self, input_size, num_hidden_layers_gen, num_dimensions_gen, num_hidden_layers_disc, num_dimensions_disc, icgan_model_path):
        self.input_shape = input_size
        self.num_hidden_layers_gen = num_hidden_layers_gen
        self.num_dimensions_gen = num_dimensions_gen
        self.num_hidden_layers_disc = num_hidden_layers_disc
        self.num_dimensions_disc = num_dimensions_disc
        self.icgan_model_path = icgan_model_path

        # Initialize networks
        self.generator = Generator(self.input_shape, num_hidden_layers_gen, num_dimensions_gen)
        self.discriminator = Discriminator(self.input_shape, num_hidden_layers_disc, num_dimensions_disc)
        # random_seed = 123
        # torch.manual_seed(random_seed)


    def train(self, batch_size, training_data, n_epochs, learning_rate_generator, learning_rate_discriminator):


        # prepare training data
        self.training_data = training_data
        self.training_data_tensor = torch.tensor(self.training_data.iloc[:,:].values, dtype=torch.float32)

        data_loader = torch.utils.data.DataLoader(
            self.training_data_tensor, batch_size=batch_size, shuffle=True
        )

        # Initialize BCE loss
        loss_function = nn.BCEWithLogitsLoss()
        real_samples_tensor = torch.ones((batch_size, self.input_shape))
        fake_samples_tensor = torch.zeros((batch_size, self.input_shape))
        # Initialize optimizers
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate_discriminator)
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=learning_rate_generator)
        self.loss_generator = 0
        self.loss_discriminator = 0
        log= []
        print(f'Training classical GAN for {n_epochs} epochs')
        
        for epoch in tqdm(range(n_epochs)):
            for n, real_samples in enumerate(data_loader):
                # Data for training the discriminator
                if len(real_samples) != batch_size:
                    continue
                else:
                    # Training Discriminator
                    self.discriminator.zero_grad()
                    real_discriminator_loss = loss_function(
                        self.discriminator(real_samples), real_samples_tensor)

                    latent_space_samples = torch.randn((batch_size, self.input_shape))
                    fake_generated_samples = self.generator(latent_space_samples)

                    fake_discriminator_loss = loss_function(
                        self.discriminator(fake_generated_samples), fake_samples_tensor)

                    #  Calculate discriminator loss = fake loss + real loss
                    self.loss_discriminator = real_discriminator_loss + fake_discriminator_loss
                    self.loss_discriminator.backward()
                    self.optimizer_discriminator.step()

                    # Training generator
                    self.generator.zero_grad()
                    latent_space_samples = torch.randn((batch_size, real_samples.shape[1]))
                    generated_samples = self.generator(latent_space_samples)
                    output_discriminator_generated = self.discriminator(generated_samples)
                    self.loss_generator = loss_function(
                        output_discriminator_generated, real_samples_tensor
                    )
                    self.loss_generator.backward()
                    self.optimizer_generator.step()

            # Show loss
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} Loss D.: {self.loss_discriminator} Loss G.: {self.loss_generator}")
            log.append((epoch, self.loss_discriminator, self.loss_generator))
            # Save model related files
            self.save_icgan(epoch)
        # save losses at the end of training
        self.save_loss(log)
        return self.calculate_trainable_params(self.generator), self.calculate_trainable_params(self.discriminator)

    def save_loss(self, log):
        with open(f"{self.icgan_model_path}/log.pickle", "wb") as f:
            pickle.dump((log), f)

    def save_icgan(self, epoch):
        # torch.save({'epoch': epoch,
        #             'model_state_dict': self.generator.state_dict(),
        #             'optimizer_state_dict': self.optimizer_generator.state_dict(),
        #             'loss': self.loss_generator},
        #              f"{self.icgan_model_path}/generator-{epoch}.pth")
        # Saving only generator model for now as we only need generator for prediction
        torch.save(self.generator, f"{self.icgan_model_path}/generator-{epoch}.pth")

    def load_icgan(self, epoch):
        model = torch.load(f"{self.icgan_model_path}/generator-{epoch}.pth", weights_only=False)
        return model

    def calculate_trainable_params(self,model):
        trainable_params = sum(
            param.numel() for param in model.parameters() if param.requires_grad
        )
        return trainable_params

    def predict(self, model, num_samples):
        return model(torch.randn(num_samples, self.input_shape)).detach().numpy()

    def evaluate_icgan(self, original_data, data_spec, epochs, numerical_columns, categorical_columns, num_pipeline,
                       cat_pipeline, model_path):


        tdf = TabularDataTransformer(data_spec)
        best_kl_original_space = np.inf
        best_overall_metric = 0.0
        best_kl_epoch = 0
        best_overall_epoch = 0
        kl_list_original_space = []
        overall_metric_list = []
        it_list = []

        progress = tqdm(range(epochs))
        print("Evaluating icgan for tabular data")
        for epoch in progress:
            icgan_model = self.load_icgan(epoch)
            vanillagan_samples = self.predict(icgan_model, len(original_data))
            synthetic_data = tdf.inverse_transform_classical_gan_data(num_pipeline, cat_pipeline, vanillagan_samples,
                                                                      numerical_columns, categorical_columns)

            # kl_metric = kl_divergence_from_data_tabular(
            #     training_data,
            #     vanillagan_samples,
            #     data_spec,
            #     conversion=False
            # )
            #kl_list_original_space.append(kl_metric)
            it_list.append(epoch)
            metadata = get_metadata(original_data)
            quality_report = evaluate_quality(
                real_data=original_data,
                synthetic_data=synthetic_data,
                metadata=metadata,
                verbose=False)
            overall_metric = quality_report.get_score()
            overall_metric_list.append(overall_metric)

            # if np.float(kl_metric) < best_kl_original_space:
            #     best_kl_epoch = epoch
            #     best_kl_original_space = kl_metric
            #     best_samples_kl = vanillagan_samples

            if np.float(overall_metric) > best_overall_metric:
                best_overall_epoch = epoch
                best_overall_metric = overall_metric
                best_samples_overall = synthetic_data

                # progress.set_postfix(
                #     kl_original_space=kl_metric,
                #
                #     refresh=False,
                # )

        # kl_best = pd.DataFrame({'iteration': [int(best_kl_epoch)],
        #                         'kl_original_space': [best_kl_original_space]})
        overall_best = pd.DataFrame({'iteration': [int(best_overall_epoch)],
                                     'best_overall_metric': [best_overall_metric]})

        eval_results = pd.DataFrame(
            {
                "iteration": np.array(it_list).astype(int),
                #"kl_original_space": np.array(kl_list_original_space).astype(float),
                'overall_metric': np.array(overall_metric_list).astype(float)
            }
        )
        eval_results = eval_results.sort_values(by=["iteration"])
        eval_results.to_csv(f"{model_path}/eval_results_icgan.csv", index=False)
       # kl_best.to_csv(f"{model_path}/best_kl_results_icgan.csv", index=False)
        overall_best.to_csv(f"{model_path}/best_overall_results_icgan.csv", index=False)


        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        # ax0.set_title(f"KL={best_kl_original_space}")
        # ax0.set_xlabel('Iterations')
        # ax0.set_ylabel('KL in original space')
        # ax0.set_ylim(bottom=0, top=max(kl_list_original_space))
        # ax0.plot(eval_results.iteration, eval_results.kl_original_space)

        ax1.set_title(f"Overall Metric={best_overall_metric}")
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Pair Metric ')
        ax1.set_ylim(bottom=0, top=max(overall_metric_list))
        ax1.plot(eval_results.iteration, eval_results.overall_metric)
        plt.savefig(f"{model_path}/eval_metrics_icgan.png")

        output = {#'kl': (best_samples_kl, best_kl_epoch, best_kl_original_space),
                  'overall_metric': (best_samples_overall, best_overall_epoch, best_overall_metric)}
        return output