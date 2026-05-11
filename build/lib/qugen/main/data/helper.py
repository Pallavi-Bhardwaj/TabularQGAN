# Copyright 2023 QUTAC, BASF Digital Solutions GmbH, BMW Group, 
# Lufthansa Industry Solutions AS GmbH, Merck KGaA (Darmstadt, Germany), 
# Munich Re, SAP SE.
import pickle

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from openpyxl.reader.excel import load_workbook
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from qugen.main.visualization.tabular_plots import heatmap_tabular, joint_hist_plot_tabular, marginal_hist_plot_tabular
from qugen.main.generator.external_classical_tabular_gan import classical_ctgan, classical_copula_gan
from typing import Union, Sequence, TextIO


from sdv.evaluation.single_table import run_diagnostic
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from sdmetrics.reports.single_table import QualityReport
import pandas as pd
from fpdf import FPDF

from matplotlib.backends.backend_pdf import PdfPages

def random_angle(n):
    return np.random.rand(n) * np.pi


def kl_divergence(p, q):
    eps = 1e-6
    cost = jnp.sum( p * jnp.log((p+eps)/(q+eps)) )
    return cost


def discretized_2d_probability_distribution(data, n_bins):
    x_min, x_max = np.min(data[:,0]), np.max(data[:,0])
    y_min, y_max = np.min(data[:,1]), np.max(data[:,1])
    return np.histogram2d(data[:,0], data[:,1], bins=(n_bins,n_bins), range=[[x_min, x_max], [y_min, y_max]], normed=None, weights=None, density=None)[0]/data.shape[0]


def kl_divergence_from_data(
    training_data: np.ndarray,
    learned_data: np.ndarray,
    number_bins: int = 16, 
    bin_range: Union[Sequence[Union[float, int]], Sequence[Sequence[Union[float, int]]], None] = None,
    dimension: int = 2,
):
    """
    Calculate the KL divergence, given training and learned/generated data. 
    By default, this function expects 2D data, but this can be changed using the argument "dimension". 

    Args:
        training_data (np.ndarray): The training data with shape (num_samples, num_dimensions).
        learned_data (np.ndarray): The learned data with shape (num_samples, num_dimensions).
        number_bins (int): The number of bins per dimension, i.e. the total number of D-dimensional bins is number_bins**dimension.
        bin_range Sequence[Union[float, int]] or Sequence[Sequence[Union[float, int]]]: The bin range, either specified for all axis with a single sequence or a sequence of bin-ranges for each individual dimension.
                  By default, the bin_range in each dimension is calculated from the min/max of the training_data.
        dimension (int): The dimensionality of the dataset.

    Returns:
        float: The KL-divergence.

    """
    training_data = training_data[:, :dimension]
    learned_data = learned_data[:, :dimension]
    if bin_range is None:
        b_ranges = [(training_data[:, i].min(), training_data[:, i].max()) for i in range(dimension)]
    elif isinstance(bin_range[0], int) or isinstance(bin_range[0], float):
        b_ranges = [bin_range for _ in range(dimension)]
    else:
        b_ranges = bin_range
    trained_histogram_np = np.histogramdd(training_data, bins=number_bins, range=b_ranges)
    learned_histogram_np = np.histogramdd(learned_data, bins=number_bins, range=b_ranges)
    train_probability = trained_histogram_np[0]/np.sum(trained_histogram_np[0])
    learned_probability = learned_histogram_np[0]/np.sum(learned_histogram_np[0])
    return kl_divergence(train_probability, learned_probability)

def convert_from_bitstring(data, bitstring_spec):
        # For in input bitstring 10101010100, 
        #  with bit string spec: [n7, c2,c3] 
        #  output is [47, 1.0, 0.0, 1.0, 0.0, 0.0 ]
        index_list = []
        for ds in bitstring_spec:
            index_list.append(int(ds[1:]))
        cum_index_list = np.cumsum(index_list)
        data_spilt = np.split(data, cum_index_list, axis =1)
        data_blocks = []
        for idx, spec in enumerate(bitstring_spec):
            data_bloc = data_spilt[idx]
            if spec[0] == 'n':
                
                func = lambda x: np.array([int("".join(map(str, x)), 2)])
                data_bloc = np.apply_along_axis(func, 1, data_bloc)
            else:
                data_bloc = data_bloc.astype(float)

            data_blocks.append(data_bloc)

        return np.concatenate(data_blocks, axis = 1)


def kl_divergence_from_data_tabular(
        training_data: np.ndarray,
        learned_data: np.ndarray,
        bitstring_spec,
        conversion = True
        ):

    b_ranges = []
    if conversion:
        train_probability_converted = convert_from_bitstring(training_data, bitstring_spec)
        sample_converted = convert_from_bitstring(learned_data, bitstring_spec)
    else:
        train_probability_converted = training_data
        sample_converted = learned_data
    bins = []
    b_ranges = []
    for spec in bitstring_spec:
        if spec[0] == 'n':
            bin_count = 2**int(spec[1:])
            bins.append(bin_count)
            b_ranges.append((0, bin_count))
            # note this will only work for numeric data which are positive integers. 
        else:
            for _ in range(int(spec[1:])):
                bins.append(2)
                b_ranges.append((0, 1))
    

    trained_histogram_np = np.histogramdd(train_probability_converted, bins=bins, range=b_ranges)
    learned_histogram_np = np.histogramdd(sample_converted, bins=bins, range=b_ranges)
    train_probability = trained_histogram_np[0]/np.sum(trained_histogram_np[0])
    learned_probability = learned_histogram_np[0]/np.sum(learned_histogram_np[0])
    return kl_divergence(train_probability, learned_probability)
    


def kl_divergence_from_data_3d(training_data: np.ndarray, learned_data: np.ndarray, number_bins=16, bin_range=[[0, 1], [0, 1], [0, 1]]):
    trained_histogram_np = np.histogramdd(training_data, bins=(number_bins, number_bins, number_bins), range=bin_range)
    learned_histogram_np = np.histogramdd(learned_data, bins=(number_bins, number_bins, number_bins), range=bin_range)
    #trained_histogram = plt.hist2d(training_data[:, 0], training_data[:, 1], bins=(number_of_bins, number_of_bins), range=[[0, 1], [0, 1]])
    #learned_histogram = plt.hist2d(learned_data[:, 0], learned_data[:, 1], bins=(number_of_bins, number_of_bins), range=[[0, 1], [0, 1]])
    train_probability = trained_histogram_np[0]/np.sum(trained_histogram_np[0])
    learned_probability = learned_histogram_np[0]/np.sum(learned_histogram_np[0])
    return kl_divergence(train_probability, learned_probability)


# Convenient plotting
def plot_samples(data, title, size=(5, 4), x_label='x', y_label='y', constrained=True):
    plt.rcParams["figure.figsize"] = size
    plt.scatter(data[:, 0], data[:, 1], s=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    if constrained:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
    plt.show()


def create_histogram_marginal_plot(data, number_bins):
    """ Create 2-D histogram with marginal histogram on the x/y axix
        Recipe: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
    """
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.04

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # 2d histogram
    ax.hist2d(data[:, 0], data[:, 1], bins=(number_bins, number_bins), range=[[0, 1], [0, 1]])
    ax_histx.hist(data[:, 0], bins=number_bins, range=[0, 1], density=False)
    ax_histy.hist(data[:, 1], bins=number_bins, range=[0, 1], orientation='horizontal', density=False)
    return fig


class CustomDataset:
    def __init__(self,data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        pass


    @property
    def data(self):
        return self._data

    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
            
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]


def plot_tabular_maps(original_data, synthetic_data, data_spec, plots_path,
                      features_names, cat_features, numerical_features, numerical_qubits, reload, evaluation):

    # plot marginal distributions for features
    for idx, feature in enumerate(features_names):
        marginal_hist_plot_tabular(original_data, synthetic_data, feature, plots_path,
                                                       data_spec[idx], reload, evaluation)

    # plot histogram between cat and continous features
    for idx, numerical_feature in enumerate(numerical_features):
        for cat_feature in cat_features:
            joint_hist_plot_tabular(original_data, synthetic_data, numerical_feature,
                                                      numerical_qubits[idx], cat_feature, plots_path, reload, evaluation)

    # plot heatmap between categorical features
    for idx, first_cat_feature in enumerate(cat_features[:-1]):
        for second_cat_feature in cat_features[idx + 1:]:
            heatmap_tabular(original_data, synthetic_data, first_cat_feature, second_cat_feature,
                                           plots_path, reload, evaluation)
    # plot heatmap between numerical features
    for idx, first_numerical_feature in enumerate(numerical_features[:-1]):
        for second_numerical_feature in numerical_features[idx + 1:]:
            heatmap_tabular(original_data, synthetic_data, first_numerical_feature, second_numerical_feature,
                                           plots_path, reload, evaluation)


def tabular_feature_names(column_names, data_spec):
    features_names = []
    cat_features = []
    numerical_features = []
    numerical_qubits = []
    input_size = 0
    for idx, spec in enumerate(data_spec):
        column = column_names[idx]
        if spec[0] == 'n':
            feature_name = column
            numerical_features.append(feature_name)
            numerical_qubits.append(int(spec[1:]))
            input_size += 1
        else:
            feature_name = list(column.keys())[0]
            cat_features.append(feature_name)
            input_size += int(spec[1:])
        features_names.append(feature_name)
    return  features_names,  input_size, cat_features, numerical_features, numerical_qubits


def plot_training_loss(file_path, reload):
    #meta_string = f"{num_used_rows}_{data_set_name}_{n_epochs}"

    # plot the log value against iterration
    log_file = file_path + 'log.pickle'
    with open(log_file, "rb") as file:
        log = pickle.load(file)
    log_array = np.array(log)
    loss_generator = log_array[:, 0]
    loss_discriminator = log_array[:, 1]

    # loss_generator, loss_discriminator  = log[0]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(range(len(loss_generator)), loss_generator, label='Generator Loss')
    ax.plot(range(len(loss_discriminator)), loss_discriminator, label='Discriminator Loss')
    ax.set_title("Training Loss")
    ax.legend()
    if reload:
        fig.savefig(f"{file_path} training_loss_reload.pdf")
    else:
        fig.savefig(f"{file_path} training_loss.pdf")
    plt.close()

def get_metadata(original_data):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(original_data)
    return metadata


def metrics_report(original_data, generated_data, metadata, model_name):

    quality_report = evaluate_quality(
        real_data=original_data,
        synthetic_data=generated_data,
        metadata=metadata)

    col_shape = quality_report.get_details(property_name="Column Shapes")
    col_trends = quality_report.get_details(property_name="Column Pair Trends")
    overall_score = quality_report.get_score()

    col_shape.rename(columns={"Score": f"{model_name}_score"}, inplace=True)
    col_trends.rename(columns={"Score": f"{model_name}_score"}, inplace=True)

    return col_shape.round(3), col_trends.round(3), overall_score.round(3)

def run_external_classical_tabular(original_data, metadata, model_type, epochs, generator_lr, discriminator_lr, batch_size,
                                   input_size, num_hidden_layers_gen,num_dimensions_gen, num_hidden_layers_disc,
                                   num_dimensions_disc):


    if model_type == 'ctgan':
        samples, params = classical_ctgan(original_data, epochs, metadata, generator_lr, discriminator_lr, batch_size,
                                          input_size,num_hidden_layers_gen,num_dimensions_gen, num_hidden_layers_disc,
                                           num_dimensions_disc)
    if model_type == 'copulagan':
        samples, params = classical_copula_gan(original_data, epochs, metadata, generator_lr, discriminator_lr, batch_size,
                                               input_size, num_hidden_layers_gen,num_dimensions_gen, num_hidden_layers_disc,
                                           num_dimensions_disc)
    print(f"{model_type} params: {params}")

    col_shapes, col_pair, overall_score = metrics_report(original_data, samples,metadata, model_type)

    return samples, col_shapes, col_pair, overall_score




def metrics_with_classical_benchmark(original_data, quantum_generated_data, path_to_models, epochs, generator_lr,
                                     discriminator_lr, batch_size, vanilla_gan_score, input_size, num_hidden_layers_gen,
                                     num_dimensions_gen, num_hidden_layers_disc, num_dimensions_disc,
                                     external_classical_benchmark):

    # create file name
    file_name = 'evaluation_metrics.pdf'
    pdf_file_path = f"{path_to_models}{file_name}"

    ctgan_overall_score = 0.0
    copulagan_overall_score = 0.0
    # convert to correct data type for numeric data
    for numeric_col in ['AGE', 'age', 'INTAKE TIME', 'intake time']:
        if numeric_col in original_data.columns:
            original_data.loc[:, numeric_col] = original_data.loc[:, numeric_col].astype('float')


    with PdfPages(pdf_file_path) as metrics_pdf:

        metrics_pdf.keep_empty = True
        # create metadat file for classical gans
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(original_data)

        # write quantum gan metrics
        quantumgan_col_shapes, quantumgan_col_pair, quantumgan_overall_score = metrics_report(original_data,
                                                                                              quantum_generated_data,
                                                                                              metadata, "quantum_gan")

        if external_classical_benchmark:


            ctgan_samples, ctgan_col_shapes, ctgan_col_pair, ctgan_overall_score = run_external_classical_tabular(original_data,
                 metadata,'ctgan',epochs,generator_lr,discriminator_lr,batch_size,input_size, num_hidden_layers_gen,
                                                    num_dimensions_gen, num_hidden_layers_disc, num_dimensions_disc)

            copulagan_samples, copulagan_col_shapes,copulagan_col_pair, copulagan_overall_score = run_external_classical_tabular(
                original_data, metadata, 'copulagan', epochs, generator_lr, discriminator_lr, batch_size, input_size,
                num_hidden_layers_gen, num_dimensions_gen, num_hidden_layers_disc, num_dimensions_disc)

            classical_metrics_col_shapes = pd.concat(
                [ctgan_col_shapes[["Column", "Metric", "ctgan_score"]], copulagan_col_shapes[["copulagan_score"]]], axis=1)

            classical_metrics_col_pairs = pd.concat(
                [ctgan_col_pair[["Column 1", "Column 2", "Metric", "ctgan_score"]],
                 copulagan_col_pair[["copulagan_score"]]], axis=1)

            col_shape_metric_scores = pd.concat(
                [classical_metrics_col_shapes[["Column", "Metric", "ctgan_score", "copulagan_score"]],
                  quantumgan_col_shapes[["quantum_gan_score"]]], axis=1)

            col_pair_metric_scores = pd.concat(
                [classical_metrics_col_pairs[["Column 1", "Column 2","Metric", "ctgan_score", "copulagan_score"]],
                  quantumgan_col_pair[["quantum_gan_score"]]], axis=1)

            ctgan_samples.to_csv( path_to_models+ "/" + "ctgan_samples" +  ".csv")
            copulagan_samples.to_csv(path_to_models + "/" + "copulagan_samples" +  ".csv")

            save_text(metrics_pdf, ctgan_overall_score, copulagan_overall_score, vanilla_gan_score,
                      quantumgan_overall_score)
                      #, ctgan_params, copulagan_params, quantum_gen_params,
                      #quantum_disc_params)
        else:
            col_shape_metric_scores = pd.concat([quantumgan_col_shapes[["Column", "Metric", "quantum_gan_score"]]], axis=1)

            col_pair_metric_scores = pd.concat([quantumgan_col_pair[["Column 1","Column 2","Metric", "quantum_gan_score"]]], axis=1)

            save_text(metrics_pdf, 0,0,0, quantumgan_overall_score)
                     # eval_quantumgan_overall_score, 0, 0, quantum_gen_params,


        write_pdf(metrics_pdf, col_pair_metric_scores, col_shape_metric_scores)
    #write_excel(col_pair_metric_scores, col_shape_metric_scores, ctgan_overall_score, copulagan_overall_score, quantumgan_overall_score,
     #              eval_quantumgan_overall_score, ctgan_params, copulagan_params, quantum_gen_params,quantum_disc_params)

def write_excel(col_pair_metric_scores, col_shape_metric_scores, ctgan_overall_score, copulagan_overall_score, quantumgan_overall_score,
                      eval_quantumgan_overall_score, ctgan_params, copulagan_params, quantum_gen_params,
                      quantum_disc_params):
    with pd.ExcelWriter('evaluation_metrics.xlsx') as writer:
        writer.write(f"ctgan score:{ctgan_overall_score}\n copulagan score:{copulagan_overall_score} \n quantumgan score:{quantumgan_overall_score} \n quantum_eval_gan_score:{eval_quantumgan_overall_score}")
        writer.write(
             f"ctgan_params:{ctgan_params}\n copulagan_params:{copulagan_params} \n quantum_gen_params:{quantum_gen_params} , quantum_disc_params:{quantum_disc_params}")
        col_pair_metric_scores.to_excel(writer, sheet_name='column_pair_metrics')
        col_shape_metric_scores.to_excel(writer, sheet_name='column_shape_metrics')


def write_pdf(metrics_pdf , col_pair_metric_scores, col_shape_metric_scores):

    save_table(col_shape_metric_scores.columns, metrics_pdf, col_shape_metric_scores)
    save_table(col_pair_metric_scores.columns, metrics_pdf, col_pair_metric_scores)

def save_table(header, pdf, table):
    table = np.asarray(table)
    fig = plt.figure(figsize=(14,14))
    ax = plt.Axes(fig, [0., 0., 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    # if score:
    #     table = np.vstack([table, [ctgan_score, copula_gan_score, quantum_gan_score]])
    tab = plt.table(cellText=table, colLabels=header, cellLoc='center', loc='center')

    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(0.9, 3.5)

    pdf.savefig(fig)

def save_text(pdf, ctgan_score, copula_gan_score, vanilla_gan_score, quantum_gan_score):
              #, ctgan_params ,copulagan_params, quantum_gen_params, quantum_disc_params):
    #fig = plt.figure(figsize=(2, 2))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    #ax = plt.Axes(fig)
    ax.set_axis_off()
    #fig.add_axes(ax)

    text1 = plt.text(0.0, 0.7,
                 f"ctgan score:{ctgan_score}\n copulagan score:{copula_gan_score}\n vanillagan score:{vanilla_gan_score} "
                 f" \n quantumgan score:{quantum_gan_score}",
                 transform=fig.transFigure, size=10)

    # text2 = plt.text(12.0, 0.7,
    #             f"ctgan_params:{ctgan_params}\n copulagan_params:{copulagan_params} \n quantum_gen_params:{quantum_gen_params} , quantum_disc_params:{quantum_disc_params}",
    #              transform=fig.transFigure, size=10)

    #text.auto_set_fontsize(True)
    #text.set_fontsize(5)
    pdf.savefig(fig)
