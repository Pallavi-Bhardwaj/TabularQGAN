

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

def marginal_hist_plot_tabular(training_data,generated_data, feature_name, file_path, spec, reload, evaluation):

    plt.rcParams["figure.figsize"] = [10.00, 9.00]
    plt.rcParams["figure.autolayout"] = True
    f, axes = plt.subplots(1, 2, sharex='all', sharey='all')

    if spec[0] == 'n':
        num_bins = 2**int(spec[1:])

        sns.histplot(training_data[[feature_name]], x=feature_name,  ax=axes[0],binwidth= 1, binrange=[0,num_bins])
        sns.histplot(generated_data[[feature_name]], x=feature_name, ax=axes[1], binwidth= 1, binrange=[0,num_bins])
    else:
        # print(generated_data[[feature_name]].value_counts())
        sns.histplot(training_data[[feature_name]], x=feature_name, ax=axes[0])
        sns.histplot(generated_data[[feature_name]], x=feature_name, ax=axes[1])
    if reload:
        f.savefig(f"{file_path} marginal_hist_plot_{feature_name}_reload.pdf")
    elif evaluation:
        f.savefig(f"{file_path} marginal_hist_plot_{feature_name}_evaluation.pdf")
    else:
        f.savefig(f"{file_path} marginal_hist_plot_{feature_name}.pdf")
    plt.close()

def joint_hist_plot_tabular(training_data,generated_data, x_feature, x_qubits, hue_feature, file_path, reload, evaluation):

    plt.rcParams["figure.figsize"] = [10.00, 9.00]
    plt.rcParams["figure.autolayout"] = True
    f, axes = plt.subplots(1, 2, sharex='all', sharey='all')
    num_bins = 2 ** x_qubits

    sns.histplot(training_data[[x_feature, hue_feature]], x=x_feature, hue=hue_feature, element="poly", ax=axes[0], binwidth= 1, binrange=[0,num_bins])
    sns.histplot(generated_data[[x_feature, hue_feature]], x=x_feature, hue=hue_feature, element="poly", ax=axes[1],  binwidth= 1, binrange=[0,num_bins])
    if reload:
        f.savefig(f"{file_path} joint_hist_plot_{x_feature}_{hue_feature}_reload.pdf")
    elif evaluation:
        f.savefig(f"{file_path} joint_hist_plot_{x_feature}_{hue_feature}_evaluation.pdf")
    else:
        f.savefig(f"{file_path} joint_hist_plot_{x_feature}_{hue_feature}.pdf")
    plt.close()

def heatmap_tabular(training_data,generated_data, x_feature, y_feature, file_path, reload, evaluation):
    plt.rcParams["figure.figsize"] = [10.00, 7.00]
    plt.rcParams["figure.autolayout"] = True
    f, axes = plt.subplots(1, 2 , sharex='all', sharey='all')

    training_df = pd.pivot_table(training_data[[ x_feature, y_feature]], columns=[x_feature],
                                                  index=[y_feature], aggfunc=len) 
    generated_df = pd.pivot_table(generated_data[[ x_feature, y_feature]], columns=[x_feature],
                                                   index=[y_feature], aggfunc=len) 
    # training_df = training_df.apply(lambda x: x/len(training_data))
    
    # generated_df = generated_df.apply(lambda x: x/len(training_data))
    sns.heatmap(training_df/len(training_data), annot=True, fmt=".2f", ax=axes[0], cmap='rocket_r')
    sns.heatmap(generated_df/len(training_data), annot=True, fmt=".2f", ax=axes[1], cmap='rocket_r')
    if reload:
        f.savefig(f"{file_path} heatmap_{x_feature}_{y_feature}_reload.pdf")
    elif evaluation:
        f.savefig(f"{file_path} heatmap_{x_feature}_{y_feature}_evaluation.pdf")
    else:
        f.savefig(f"{file_path} heatmap_{x_feature}_{y_feature}.pdf")
    plt.close()