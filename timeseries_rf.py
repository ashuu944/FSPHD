import configuration as conf
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump
from visualization import save_classification_map, save_region_classification_map, plot_confusion_matrix, plot_roc_auc, plot_regression_results

# Load data function
def load_data(rep, r_split, country):
    # Load data
    X_train = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "timeseries_x_train.npy"))
    X_test = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "timeseries_x_test.npy"))
    y_train = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_" + rep + "/y_train.npy"))
    y_test = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_" + rep + "/y_test.npy"))
    info_train = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "info_train.npy"))
    info_test = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "info_test.npy"))
    return X_train, X_test, y_train, y_test, info_train, info_test

# Train and evaluate model functions
def train_model(algorithm, model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(algorithm, model, X_test, y_test):
    predictions = model.predict(X_test)
    if algorithm == 'classification':
        score = accuracy_score(y_test, predictions)
    else:
        score = r2_score(y_test, predictions)
    return predictions, score

# Save model
def save_model(country, rep, tt_split, algorithm, model):
    model_dir = os.path.join(conf.OUTPUT_DIR, country, "models", "random_forest", tt_split, algorithm, rep)
    os.makedirs(model_dir, exist_ok=True)
    dump(model, os.path.join(model_dir, 'random_forest_model.joblib'))

from sklearn.preprocessing import label_binarize

def timeseries_random_forest(rep, algorithm, r_split, country, tt_split):
    X_train, X_test, y_train, y_test, info_train, info_test = load_data(rep, r_split, country)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if algorithm == 'classification':
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_probabilities = model.predict_proba(X_test)
    else:
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_probabilities = predictions  # For regression, test_probabilities is just predictions

    save_model(country, rep, tt_split, algorithm, model)
    save_results(country, algorithm, rep, tt_split, y_test, predictions, info_test, test_probabilities)

def save_results(country, algorithm, rep, tt_split, test_targets, test_predictions, y_test_com, test_probabilities):
    try:
        # Ensure y_test_com[:, 0] and y_test_com[:, 1] are lists of one-dimensional data
        temporal_data = y_test_com[:, 0]
        region_data = y_test_com[:, 1]

        # Check if these are 1D arrays
        if temporal_data.ndim != 1 or region_data.ndim != 1:
            raise ValueError("temporal_data or region_data are not 1-dimensional.")

        #print(f"temporal_data shape: {temporal_data.shape}, region_data shape: {region_data.shape}")
        #print(f"test_targets shape: {np.array(test_targets).shape}, test_predictions shape: {np.array(test_predictions).shape}")

        # Ensure test_targets and test_predictions are 1D
        test_targets = np.array(test_targets).flatten()
        test_predictions = np.array(test_predictions).flatten()

        # Ensure all data is 1-dimensional
        if test_targets.ndim != 1 or test_predictions.ndim != 1:
            raise ValueError("test_targets or test_predictions are not 1-dimensional.")

        output = pd.DataFrame({
            conf.TEMPORAL_GRANULARITY[country]: temporal_data.tolist(),
            conf.ID_REGIONS[country].upper(): region_data.tolist(),
            'label': test_targets,
            'prediction': test_predictions
        })
        
        # Proceed with merging and saving results
        data = pd.read_excel(os.path.join(conf.DATA_DIRECTORY, country, conf.RESPONSE_FILE[country]))
        results = pd.merge(output, data[[conf.TEMPORAL_GRANULARITY[country], conf.SPATIAL_GRANULARITY[country][-1], conf.SPATIAL_GRANULARITY[country][-2],
                              conf.ID_REGIONS[country].upper()]], on= [conf.TEMPORAL_GRANULARITY[country], conf.ID_REGIONS[country].upper()], how='inner')
        rearranged_columns = [conf.TEMPORAL_GRANULARITY[country], conf.SPATIAL_GRANULARITY[country][-1], conf.SPATIAL_GRANULARITY[country][-2],
                              conf.ID_REGIONS[country].upper(), 'label', 'prediction']
        results = results[rearranged_columns]
        os.makedirs(os.path.join(conf.OUTPUT_DIR, country, "results", "random_forest", tt_split, algorithm), exist_ok=True)
        results.to_excel(os.path.join(conf.OUTPUT_DIR, country, "results", "random_forest", tt_split, algorithm,  rep + '.xlsx'), index=False)

        if algorithm == 'classification':
            save_classification_map(country, algorithm, tt_split, "random_forest", rep, max(y_test_com[:, 0].tolist()))
            save_region_classification_map(country, algorithm, tt_split, "random_forest", rep, max(y_test_com[:, 0].tolist()))
            plot_confusion_matrix(country, algorithm, tt_split, "random_forest", rep, max(y_test_com[:, 0].tolist()))
            plot_roc_auc(country, algorithm, tt_split, "random_forest", rep, max(y_test_com[:, 0].tolist()), test_probabilities)
        else:
            plot_regression_results(country, algorithm, tt_split, "random_forest", rep, max(y_test_com[:, 0].tolist()))
    except ValueError as e:
        print(f"Error while saving results: {e}")