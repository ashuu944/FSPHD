import configuration as conf
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump
from visualization import save_classification_map, save_region_classification_map, plot_confusion_matrix, plot_roc_auc, plot_regression_results

def load_features(country, algorithm, rep,r_split, tt_split):
    # Define the base directory for features
    base_dir = os.path.join(conf.OUTPUT_DIR, country, "best_features", algorithm, tt_split, rep)
    
    # Load LSTM features
    lstm_train = np.load(os.path.join(base_dir, 'lstm_feat_X_train.npy'))
    lstm_test = np.load(os.path.join(base_dir, 'lstm_feat_X_test.npy'))
    
    # Load CNN features
    #cnn_train = np.load(os.path.join(base_dir, 'cnn_feat_X_train.npy'))
    #cnn_test = np.load(os.path.join(base_dir, 'cnn_feat_X_test.npy'))
    
    # Load combined conjunctural and spatial features
    cs_train = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cs_x_train.npy"))
    cs_test = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cs_x_test.npy"))
    
    # Load info data
    info_train = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "info_train.npy"))
    info_test = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "info_test.npy"))

    # Load target labels
    y_train = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, f"features_{rep}/y_train.npy"))
    y_test = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, f"features_{rep}/y_test.npy"))

   
    print("LSTM Train Shape:", lstm_train.shape)

    #print("CNN Train Shape:", cnn_train.shape)
    print("CS Train Shape:", cs_train.shape)
    print("LSTM Test Shape:", lstm_test.shape)
    #print("CNN Test Shape:", cnn_test.shape)
    print("CS Test Shape:", cs_test.shape)
    print("info Train Shape:", info_train.shape)
    print("innfo Test Shape:", cs_test.shape)
    print("y Train Shape:", y_train.shape)
    print("y Test Shape:", y_test.shape)
    

    # Ensure all features are 2-dimensional
    lstm_train = np.atleast_2d(lstm_train)
    lstm_test = np.atleast_2d(lstm_test)
    cnn_train = np.atleast_2d(cnn_train)
    cnn_test = np.atleast_2d(cnn_test)
    cs_train = np.atleast_2d(cs_train)
    cs_test = np.atleast_2d(cs_test)

    
    # Concatenate LSTM, CNN, and CS (conjunctural + spatial) features
    X_train = np.concatenate((lstm_train, cnn_train, cs_train), axis=0)
    X_test = np.concatenate((lstm_test, cnn_test, cs_test), axis=0)
    
    # Concatenate CNN, and CS (conjunctural + spatial) features
    #X_train = np.concatenate((cnn_train, cs_train), axis=1)
    #X_test = np.concatenate((cnn_test, cs_test), axis=1)
    
   

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
def save_rf_model(country, rep, tt_split, algorithm, model):
    model_dir = os.path.join(conf.OUTPUT_DIR, country, "models", "rf", tt_split, algorithm, rep)
    os.makedirs(model_dir, exist_ok=True)
    dump(model, os.path.join(model_dir, 'rf_fusion_model.joblib'))

# Save results
def save_rf_results(country, algorithm, rep, tt_split, test_targets, test_predictions, y_test_com, test_probabilities):
    try:
        temporal_data = y_test_com[:, 0]
        region_data = y_test_com[:, 1]
        test_targets = np.array(test_targets).flatten()
        test_predictions = np.array(test_predictions).flatten()

        output = pd.DataFrame({
            conf.TEMPORAL_GRANULARITY[country]: temporal_data.tolist(),
            conf.ID_REGIONS[country].upper(): region_data.tolist(),
            'label': test_targets,
            'prediction': test_predictions
        })
        
        data = pd.read_excel(os.path.join(conf.DATA_DIRECTORY, country, conf.RESPONSE_FILE[country]))
        results = pd.merge(output, data[[conf.TEMPORAL_GRANULARITY[country], conf.SPATIAL_GRANULARITY[country][-1], conf.SPATIAL_GRANULARITY[country][-2],
                              conf.ID_REGIONS[country].upper()]], on= [conf.TEMPORAL_GRANULARITY[country], conf.ID_REGIONS[country].upper()], how='inner')
        rearranged_columns = [conf.TEMPORAL_GRANULARITY[country], conf.SPATIAL_GRANULARITY[country][-1], conf.SPATIAL_GRANULARITY[country][-2],
                              conf.ID_REGIONS[country].upper(), 'label', 'prediction']
        results = results[rearranged_columns]
        os.makedirs(os.path.join(conf.OUTPUT_DIR, country, "results", "rf", tt_split, algorithm), exist_ok=True)
        results.to_excel(os.path.join(conf.OUTPUT_DIR, country, "results", "rf", tt_split, algorithm,  rep + '.xlsx'), index=False)

        if algorithm == 'classification':
            save_classification_map(country, algorithm, tt_split, "rf", rep, max(y_test_com[:, 0].tolist()))
            save_region_classification_map(country, algorithm, tt_split, "rf", rep, max(y_test_com[:, 0].tolist()))
            plot_confusion_matrix(country, algorithm, tt_split, "rf", rep, max(y_test_com[:, 0].tolist()))
            plot_roc_auc(country, algorithm, tt_split, "rf", rep, max(y_test_com[:, 0].tolist()), test_probabilities)
        else:
            plot_regression_results(country, algorithm, tt_split, "rf", rep, max(y_test_com[:, 0].tolist()))
    except ValueError as e:
        print(f"Error while saving results: {e}")

def feature_fusion_rf(rep, algorithm, r_split, country, tt_split):
    X_train, X_test, y_train, y_test, info_train, info_test = load_features(country, algorithm, rep, r_split, tt_split)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if algorithm == 'classification':
        model = RandomForestClassifier(n_estimators=900,max_depth=20,random_state=1)
    else:
        model = RandomForestRegressor(n_estimators=900,max_depth=20,random_state=1)
    
    model = train_model(algorithm, model, X_train, y_train)
    predictions, score = evaluate_model(algorithm, model, X_test, y_test)
    test_probabilities = model.predict_proba(X_test) if algorithm == 'classification' else predictions

    save_rf_model(country, rep, tt_split, algorithm, model)
    save_rf_results(country, algorithm, rep, tt_split, y_test, predictions, info_test, test_probabilities)

    print("End of Feature Fusion Training and Testing through RF model")
