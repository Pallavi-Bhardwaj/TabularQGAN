from os.path import split

from qugen.main.data.data_handler import TabularDataTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import  accuracy_score
import numpy as np
import pandas as pd


def train_xgb_classifier(original_data, data_spec, target_feature, numerical_columns, categorical_columns, model_path,
                         filename, train):

    model, task = initialize_xgboost_model(categorical_columns, numerical_columns, target_feature)


    tdf = TabularDataTransformer(data_spec)

    target = pd.DataFrame(original_data[f'{target_feature}'])
    features = original_data.loc[:, original_data.columns != f'{target_feature}']

    training_features, training_labels = process_features_and_target(categorical_columns, features, numerical_columns,
                                                                     target, tdf)

    # Split original data into train and test sets
    features_test, features_train, label_test, label_train = split_train_test(numerical_columns, target_feature,
                                                                              training_features, training_labels)

    if train:
        # Define parameter grid
        param_grid = {
            'max_depth': [2, 3, 4, 5],
            'learning_rate': [0.1, 0.01, 0.001, 0.05],
            'n_estimators': [50, 100, 200],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Perform grid search for original data
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=1)
        grid_search.fit(features_train, label_train)

        # Print best score and parameters
        print(f"Best score for original data: {grid_search.best_score_:.3f}")
        print(f"Best parameters for original data: {grid_search.best_params_}")


        # Access best model
        model = grid_search.best_estimator_

        # Save best model
        model.save_model(f"{model_path}/xgboost_{filename}.json")
    else:
        # # Load saved model
        # if task == 'regression':
        #     model = XGBRegressor()
        # elif task == 'classification':
        #     model = XGBClassifier()

        model.load_model(f"{model_path}/xgboost_{filename}.json")

    # Use loaded model for predictions
    predictions = model.predict(features_test)

    # Calculate accuracy and mae
    mae = mean_absolute_error(label_test, predictions)
    r2 = r2_score(label_test, predictions)
    accuracy = None

    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"R2 Score: {r2:.3f}")

    if task == 'classification':
       # f1_score = f1_score(label_test, predictions, average='micro')
        accuracy = accuracy_score(label_test, predictions)
        print(f"accuracy Score: {accuracy:.3f}")
        #metrics = pd.DataFrame({'Model': [filename],'Task type': [task] ,'Accuracy': [accuracy],'Mean Absolute Error': [mae], 'R2 Score': [r2], 'Target': [target_feature], 'features': [features.columns.tolist()], 'Best parameters for grid search': [grid_search.best_params_]})
        metrics = [filename, task, accuracy, mae, r2, target_feature, features.columns.tolist(),grid_search.best_params_]
    else:
       # metrics = pd.DataFrame({'Model': [filename], 'Task type': [task] ,'Mean Absolute Error': [mae], 'R2 Score': [r2], 'Target': [target_feature], 'features': [features.columns.tolist()], 'Best parameters for grid search': [grid_search.best_params_]})
        metrics = [filename, task,'NA', mae, r2, target_feature, features.columns.tolist(),grid_search.best_params_]

    # Save all metrics to csv
    #metrics.to_csv(f"{model_path}/xgboost_metrics_{filename}.csv")

    if task == 'classification':
        return predictions, accuracy, metrics
    else:
        return predictions, mae, metrics


def split_train_test(numerical_columns, target_feature, training_features, training_labels):
    if target_feature in numerical_columns:
        features_train, features_test, label_train, label_test = train_test_split(training_features, training_labels,
                                                                                  test_size=0.2, random_state=42)
    else:
        features_train, features_test, label_train, label_test = train_test_split(training_features, training_labels,
                                                                                  test_size=0.2, random_state=42,
                                                                                  stratify=training_labels)
    return features_test, features_train, label_test, label_train


def initialize_xgboost_model(categorical_columns, numerical_columns, target_feature):
    if target_feature in numerical_columns:
        model = XGBRegressor()
        task = 'regression'
    if target_feature in categorical_columns:
        # Create XGBClassifier
        model = XGBClassifier()
        task = 'classification'
    return model, task


def process_features_and_target(categorical_columns, features, numerical_columns, target, tdf):

    # Filter numerical and categorical columns for features and targets
    try:
        numerical_features = features[numerical_columns]
    except:
        numerical_features = []

    categorical_features = features.loc[:, features.columns != f'{numerical_columns}']
    numerical_feature_names, categorical_features_names = split_num_cat_features_and_target(features,
                                                                                            categorical_columns,
                                                                                            numerical_columns)
    numerical_target_names, categorical_target_names = split_num_cat_features_and_target(target, categorical_columns,
                                                                                         numerical_columns)

    # Transform categorical columns with OneHotEncoder
    cat_features, num_features_pipeline, cat_features_pipeline = tdf.transform_classical_gan_data(categorical_features,
                                                                                                  numerical_feature_names,
                                                                                                  categorical_features_names)
    cat_labels, num_target_pipeline, cat_target_pipeline = tdf.transform_classical_gan_data(target,
                                                                                            numerical_target_names,
                                                                                            categorical_target_names)

    if len(numerical_features) > 0:
        training_features = pd.DataFrame(
            np.concatenate([numerical_features[numerical_feature_names], cat_features], axis=1))
        training_labels = pd.DataFrame(np.concatenate([numerical_features[numerical_target_names], cat_labels], axis=1))
    else:
        training_features = pd.DataFrame(np.concatenate([cat_features], axis=1))
        training_labels = pd.DataFrame(np.concatenate([cat_labels], axis=1))

    return training_features, training_labels


def split_num_cat_features_and_target(features, categorical_columns, numerical_columns):
    numerical_features = []
    categorical_features = []
    feature_columns = features.columns
    for col in feature_columns:
        if col in numerical_columns:
             numerical_features.append(col)
        if col in categorical_columns:
            categorical_features.append(col)
    return numerical_features, categorical_features


def load_xgb_model(model_path):
    model = XGBRegressor()
    model.load_model(model_path)
    return model