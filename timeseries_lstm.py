from logger import log
import numpy as np
import configuration as conf
import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
#from torchinfo import summary
import math
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler
from visualization import save_classification_map, save_region_classification_map, plot_confusion_matrix, plot_roc_auc, plot_regression_results 
import torch.nn.functional as functional


hm_epochs = 1000
batch_size = 32

timesteps = 14
nb_hidden = 128
num_layers = 1
learning_rate = 0.01
num_classes = 3
best_test_loss = float("inf")
best_test_R2 = 0
best_ep = 0
early_stopping_patience = 100

# =============================================================================#
# Load data X, Y, and w from the preprocessed data                             #
# =============================================================================#
def load_data(rep, r_split, country):
    # Load data
    X_train = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "timeseries_x_train.npy"))
    log(country, "Loading X_train from: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "timeseries_x_train.npy"))
    X_test = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "timeseries_x_test.npy"))
    log(country, "Loading X_test from: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "timeseries_x_test.npy"))
    y_train = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_" + rep + "/y_train.npy"))
    log(country, "Loading y_train from: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_" + rep + "/y_train.npy"))
    y_test = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_" + rep + "/y_test.npy"))
    log(country,"Loading y_test from: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_" + rep + "/y_test.npy"))
    w_train = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "w_train.npy"))
    log(country, "Loading w_train from: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "w_train.npy"))
    w_test = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "w_test.npy"))
    log(country, "Loading w_test from: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "w_test.npy"))
    
    info_train = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "info_train.npy"))
    log(country, f"Loading info_train from with {info_train.shape}: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "info_train.npy"))
    info_test = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "info_test.npy"))
    log(country, f"Loading info_test from with {info_test.shape}: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "info_test.npy"))

    return X_train, X_test, y_train, y_test, w_train, w_test, info_train, info_test


# =============================================================================#
# Define LSTM architecture                                           #
# =============================================================================#



class LSTMModel(nn.Module):
    def __init__(self, input_size, algorithm):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=nb_hidden, num_layers=num_layers, batch_first=True)
       
        if algorithm == 'classification':
            #print('Classification selected')
            self.fc = nn.Linear(nb_hidden, num_classes)
        else:
            print('Regression selected')
            self.fc = nn.Linear(nb_hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


# =============================================================================#
# Training function                                                            #
# =============================================================================#
def train_model(algorithm, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    for inputs, targets, weights in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)  # Outputs should be of shape (N, C)
        targets = targets.view(-1)  # Ensure targets are of shape (N,)
        if algorithm == 'classification':
            
            adjusted_targets = targets - 1
            loss = criterion(outputs, adjusted_targets)
        else:
            loss = criterion(outputs.squeeze(), targets)
        weighted_loss = (loss * weights.view(-1)).mean()  # Ensure weights are of shape (N,)
        weighted_loss.backward()
        optimizer.step()
        running_loss += weighted_loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        if algorithm == 'classification':
            predictions = torch.argmax(outputs, dim=1).cpu().numpy() + 1
            probabilities = functional.softmax(outputs, dim=1)
        else:
            predictions = outputs.squeeze().detach().cpu().numpy()
        targets_np = targets.squeeze().detach().cpu().numpy()
        all_predictions.extend(predictions)
        all_targets.extend(targets_np)
        if algorithm == 'classification':
            all_probabilities.extend(probabilities.squeeze().detach().cpu().numpy())
    
    epoch_loss = running_loss / total_samples
    if algorithm == 'classification':
        score = accuracy_score(all_targets, all_predictions)
    else:
        score = r2_score(all_targets, all_predictions)

    return epoch_loss, score, all_targets, all_predictions, np.array(all_probabilities)



# =============================================================================#
# Evaluation function                                                          #
# =============================================================================#
def evaluate_model(algorithm, model, test_loader, criterion):
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    running_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets, weights in test_loader:
            outputs = model(inputs)
            targets = targets.view(-1)  # Ensure targets are of shape (N,)
            if algorithm == 'classification':
                adjusted_targets = targets - 1
                loss = criterion(outputs, adjusted_targets)
            else:
                loss = criterion(outputs.squeeze(), targets)
            weighted_loss = (loss * weights).mean()
            running_loss += weighted_loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            if algorithm == 'classification':
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()  + 1
                probabilities = functional.softmax(outputs, dim=1)
            else:
                predictions = outputs.squeeze().detach().cpu().numpy()
            targets_np = targets.squeeze().detach().cpu().numpy()
            all_predictions.extend(predictions)
            all_targets.extend(targets_np)
            if algorithm == 'classification':
                all_probabilities.extend(probabilities.squeeze().detach().cpu().numpy())
    
    epoch_loss = running_loss / total_samples
    if algorithm == 'classification':
        score = accuracy_score(all_targets, all_predictions)
    else:
        score = r2_score(all_targets, all_predictions)

    return epoch_loss, score, all_targets, all_predictions, np.array(all_probabilities)


# =============================================================================#
# Save model and print summary of the model                                                                #
# =============================================================================#
def save_model(country, rep, tt_split, algorithm, lstm, best_test_loss, best_test_R2, best_ep):
    os.makedirs(os.path.join(conf.OUTPUT_DIR, country, "models", "lstm", tt_split, algorithm , rep), exist_ok=True)
    torch.save({
            
            'best_ep': best_ep,
            'best_test_loss': best_test_loss,
            'model_state_dict': lstm.state_dict(),
            'best_test_R2': best_test_R2
        }, os.path.join(conf.OUTPUT_DIR, country, "models", "lstm", tt_split, algorithm, rep,'lstm_epa.pth') )
    torch.save(lstm, os.path.join(conf.OUTPUT_DIR, country, "models", "lstm", tt_split, algorithm, rep, 'lstm_epa_architecture.pth'))
    # summary(lstm, input_size=(batch_size, nb_inputs))

#Save Results
def save_results(country, algorithm, rep, tt_split, test_targets, test_predictions, y_test_com, test_probabilities):
   
    
    data = pd.read_excel(os.path.join(
        conf.DATA_DIRECTORY, country, conf.RESPONSE_FILE[country]))
    output = pd.DataFrame({conf.TEMPORAL_GRANULARITY[country]: y_test_com[:, 0].tolist(),  conf.ID_REGIONS[country].upper(): y_test_com[:, 1].tolist(), 
                           'label': test_targets, 'prediction': test_predictions})
    
    results = pd.merge(output, data[[conf.TEMPORAL_GRANULARITY[country], conf.SPATIAL_GRANULARITY[country][-1], conf.SPATIAL_GRANULARITY[country][-2],
                          conf.ID_REGIONS[country].upper()]], on= [conf.TEMPORAL_GRANULARITY[country], conf.ID_REGIONS[country].upper()], how='inner')
    rearranged_columns = [conf.TEMPORAL_GRANULARITY[country], conf.SPATIAL_GRANULARITY[country][-1], conf.SPATIAL_GRANULARITY[country][-2],
                          conf.ID_REGIONS[country].upper(), 'label', 'prediction']
    results = results[rearranged_columns]
    os.makedirs(os.path.join(conf.OUTPUT_DIR, country, "results", "lstm", tt_split, algorithm), exist_ok=True)
    results.to_excel(os.path.join(conf.OUTPUT_DIR, country, "results", "lstm", tt_split, algorithm,  rep + '.xlsx'), index=False)
    if algorithm == 'classification':
        save_classification_map(country, algorithm, tt_split, "lstm", rep, max(y_test_com[:, 0].tolist()))
        save_region_classification_map(country, algorithm, tt_split, "lstm", rep, max(y_test_com[:, 0].tolist()))
        plot_confusion_matrix(country, algorithm, tt_split, "lstm", rep, max(y_test_com[:, 0].tolist()))
        plot_roc_auc(country, algorithm, tt_split, "lstm", rep, max(y_test_com[:, 0].tolist()), test_probabilities)
    else:
        plot_regression_results(country, algorithm, tt_split, "lstm", rep, max(y_test_com[:, 0].tolist()))
# =============================================================================#
# Main function                                                                #
# =============================================================================#
def timeseries_lstm(rep, algorithm, r_split, country, tt_split):
    log(country, "Begin time-series data learning through LSTM model")

    X_train, X_test, y_train, y_test, w_train, w_test, info_train, info_test = load_data(rep, r_split, country)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long if algorithm == 'classification' else torch.float32),
        torch.tensor(w_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long if algorithm == 'classification' else torch.float32),
        torch.tensor(w_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = LSTMModel(input_size=X_train.shape[1], algorithm=algorithm)
    # Initialize model, criterion, optimizer, and scheduler
    if os.path.exists(os.path.join(conf.OUTPUT_DIR, country, "models",  "lstm", tt_split , algorithm, rep, 'lstm_epa.pth')):
        log(country, "Best LSTM Model Loaded from : "+ os.path.join(conf.OUTPUT_DIR, country, "models",  "lstm", tt_split , algorithm, rep, 'lstm_epa.pth'))
        checkpoint = torch.load(os.path.join(conf.OUTPUT_DIR, country, "models",  "lstm", tt_split , algorithm, rep, 'lstm_epa.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        best_ep = checkpoint['best_ep']
        best_test_R2 = checkpoint['best_test_R2']
        
    criterion = nn.CrossEntropyLoss() if algorithm == 'classification' else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_test_R2 = -float("inf")
    best_test_loss = float("inf")
    best_epoch = 0

    # Training loop
    for epoch in range(hm_epochs):
        # Train the model
        train_loss, train_score, train_targets, train_predictions, train_probabilities = train_model(algorithm, model, train_loader, criterion, optimizer)
        
        # Evaluate the model
        test_loss, test_score, test_targets, test_predictions, test_probabilities = evaluate_model(algorithm, model, test_loader, criterion)

        # Logging and saving best model
        log(country, f"Epoch {epoch+1}/{hm_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Train Score: {train_score:.6f}, Test Score: {test_score:.6f}")
    
        # Save the best model based on the test score
        if (algorithm == 'classification' and test_score > best_test_R2) or (algorithm != 'classification' and test_score > best_test_R2 and test_loss < best_test_loss):
            best_test_loss = test_loss
            best_test_R2 = test_score
            best_epoch = epoch + 1
            save_model(country, rep, tt_split, algorithm, model, best_test_loss, best_test_R2, best_epoch)
            save_results(country, algorithm, rep, tt_split, test_targets, test_predictions, info_test, test_probabilities)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if (epochs_no_improve >= early_stopping_patience) and best_test_R2 > 0:
            log(country, f'Early stopping at epoch {epoch+1}')
            break
        
        # Adjust learning rate scheduler
        scheduler.step(test_loss)
    log(country, "Best LSTM Model Saved at : "+ os.path.join(conf.OUTPUT_DIR, country, "models", "lstm", tt_split, algorithm, rep,'lstm_epa.pth'))
    checkpoint = torch.load(os.path.join(conf.OUTPUT_DIR, country, "models", "lstm", tt_split, algorithm, rep,'lstm_epa.pth'))
    best_ep = checkpoint['best_ep']
    best_test_R2 = checkpoint['best_test_R2']
    log(country, f"Test R2 associate with best Score: {best_test_R2:.6f}  reached at epoch: {best_ep}")
    log(country, "End time-series data learning through LSTM model")
