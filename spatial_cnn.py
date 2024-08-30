"""
Created on Tue May 12 13:20:01 2024

@author: syed
"""
import time
from logger import log
from logger import Logs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from math import ceil
from sklearn.metrics import r2_score, accuracy_score
import torch.nn.functional as F
import os
import configuration as conf
from sklearn.impute import SimpleImputer
from visualization import save_classification_map, save_region_classification_map, plot_confusion_matrix, plot_roc_auc, plot_regression_results 
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

PATH = conf.FEATURES_DIRECTORY


def next_batch(batch_size, X_data, y_data, is_train, start_index = 0):
    if is_train :
        idx = np.random.choice(len(X_data), batch_size, replace=False)
        return X_data[idx], y_data[idx], 0
    end_index = start_index + batch_size
    
    if end_index > len(X_data):
        end_index = len(X_data)
    
    batch_x = X_data[start_index:end_index]
    batch_y = y_data[start_index:end_index]
    
    new_start_index = end_index if end_index < len(X_data) else 0  # Reset to 0 if end of data is reached
    
    return batch_x, batch_y, new_start_index


class ConvNet(nn.Module):
    def __init__(self, algorithm, nbfilter1, nbfilter2, nbfilter3, shapeconv, shapepool, finalshape, L, input_channels=4):
        super(ConvNet, self).__init__()
        self.algorithm = algorithm
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=nbfilter1,
            kernel_size=shapeconv,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=nbfilter1,
            out_channels=nbfilter2,
            kernel_size=shapeconv,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=nbfilter2,
            out_channels=nbfilter3,
            kernel_size=shapeconv,
            padding=1,
        )
        
        # Adjusted pooling layer
        self.pool = nn.MaxPool2d(kernel_size=shapepool, stride=shapepool, padding=1)

        # Calculate the size of the tensor after the final pooling layer
        self._to_linear = self._get_conv_output((input_channels, L, L))
        
        self.fc1 = nn.Linear(self._to_linear, nbfilter3)
        
        if self.algorithm == 'regression':
            self.out = nn.Linear(nbfilter3, 1)
        else:
            self.out = nn.Linear(nbfilter3, 3)
        self._initialize_weights()

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        features = F.relu(self.fc1(x))
        # if self.algorithm == 'regression':
        #     outputs = self.regression_out(features)
        # else:  # classification
        outputs = self.out(features)
        return outputs, features



def load_data(rep, country):
    PATH = os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY)
    X_train = np.load(os.path.join(PATH , "cnn_x_pix_train.npy"))
    log(country, "Loading cnn_x_pix_train from: "+ os.path.join(PATH , "cnn_x_pix_train.npy"))
    X_test = np.load(os.path.join(PATH , "cnn_x_pix_test.npy"))
    log(country, "Loading cnn_x_pix_test from: "+ os.path.join(PATH , "cnn_x_pix_test.npy"))
    y_train = np.load(os.path.join(PATH , "cnn_y_pix_train.npy"))
    log(country, "Loading cnn_y_pix_train from: "+ os.path.join(PATH , "cnn_y_pix_train.npy"))
    y_test = np.load(os.path.join(PATH , "cnn_y_pix_test.npy"))
    log(country, "Loading cnn_y_pix_test from: "+ os.path.join(PATH , "cnn_y_pix_test.npy"))
    y_train_com = np.load(os.path.join(PATH , "cnn_info_pix_train.npy"))
    log(country, "Loading y_train from: "+ os.path.join(PATH , "features_" + rep , "cnn_info_pix_train.npy"))
    y_test_com = np.load(os.path.join(PATH , "cnn_info_pix_test.npy"))
    log(country, "Loading y_test from: "+ os.path.join(PATH , "features_" + rep , "cnn_info_pix_test.npy"))

    return X_train, X_test, y_train, y_test, y_train_com, y_test_com


def reshape_data(X_train, X_test, patch_size):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], patch_size, patch_size)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], patch_size, patch_size)
    return X_train, X_test


def initialize_parameters(L):
    hm_epochs = 50
    batch_size = 32
    nbfilter1 = 32
    nbfilter2 = 64
    nbfilter3 = 128
    shapeconv = 3
    shapepool = 2
    finalshape = 1,
    num_classes = 3
    early_stopping_patience = 30
    return (
        hm_epochs,
        batch_size,
        nbfilter1,
        nbfilter2,
        nbfilter3,
        shapeconv,
        shapepool,
        finalshape,
        num_classes,
        early_stopping_patience
    )


# =============================================================================#
# Save model and print summary of the model                                    #
# =============================================================================#
def save_model(country, rep, tt_split, algorithm, cnn, best_test_loss, best_test_R2, best_ep): #cnn, rep, best_test_R2, best_ep, country)
    os.makedirs(os.path.join(conf.OUTPUT_DIR, country, "models", "cnn", tt_split, algorithm, rep), exist_ok=True)
    torch.save({
            'best_ep': best_ep,
            'model_state_dict': cnn.state_dict(),
            'best_test_R2': best_test_R2
        }, os.path.join(conf.OUTPUT_DIR, country, "models",  "cnn", tt_split, algorithm, rep, 'cnn_epa.pth'))
    #torch.save(cnn.state_dict(), "./Models/cnn_epa.pth")
    torch.save(cnn, os.path.join(conf.OUTPUT_DIR, country, "models", "cnn", tt_split, algorithm, rep, 'cnn_epa_architecture.pth'))

#===========================================================================#
# Save Best Features
# ===========================================================================#
def extract_and_save_cnn_features(model, X_data, algorithm, country, rep, tt_split, data_type):
    model.eval()
    with torch.no_grad():
        _, features = model(X_data)
        features = features.cpu().numpy()
        
        # Construct the save directory based on the input parameters
        save_dir = os.path.join(conf.OUTPUT_DIR, country, "best_features", algorithm, tt_split,rep)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the features with the appropriate filename
        save_path = os.path.join(save_dir, f'cnn_feat_X_{data_type}.npy')
        np.save(save_path, features)
        log(country, f"Features saved to {save_path}")  
    

# Define a function to extract features and predictions
def extract_features_and_predictions(model, data):
    with torch.no_grad():
        features, predictions = model(data)
        features = features.cpu().numpy()
        predictions = predictions.cpu().numpy()
    return np.asarray(features, dtype=np.float32), np.asarray(predictions, dtype=np.float32)


#Save Results
def save_results(country, algorithm, rep, tt_split, test_targets, test_predictions, y_test_com, test_probabilities):
    data = pd.read_excel(os.path.join(
        conf.DATA_DIRECTORY, country, conf.RESPONSE_FILE[country]))
    output = pd.DataFrame({conf.TEMPORAL_GRANULARITY[country]: y_test_com[:, 0].tolist(),  conf.ID_REGIONS[country].upper(): y_test_com[:, 1].tolist(), 
                           'label': test_targets, 'prediction': test_predictions})
    #output['difference'] = np.where(output['label'] != output['prediction'], output['label'], np.nan)
    results = pd.merge(output, data[[conf.TEMPORAL_GRANULARITY[country], conf.SPATIAL_GRANULARITY[country][-1], conf.SPATIAL_GRANULARITY[country][-2],
                          conf.ID_REGIONS[country].upper()]], on= [conf.TEMPORAL_GRANULARITY[country], conf.ID_REGIONS[country].upper()], how='inner')
   
    rearranged_columns = [conf.TEMPORAL_GRANULARITY[country], conf.SPATIAL_GRANULARITY[country][-1], conf.SPATIAL_GRANULARITY[country][-2],
                          conf.ID_REGIONS[country].upper(), 'label', 'prediction']
    

    results = results[rearranged_columns]
    os.makedirs(os.path.join(conf.OUTPUT_DIR, country, "results", "cnn", tt_split, algorithm), exist_ok=True)
    results.to_excel(os.path.join(conf.OUTPUT_DIR, country, "results", "cnn", tt_split, algorithm,  rep + '.xlsx'), index=False)
    if algorithm == 'regression':
        plot_regression_results(country, algorithm, tt_split, "cnn", rep, max(y_test_com[:, 0].tolist()))
    else:
        save_classification_map(country, algorithm, tt_split, "cnn", rep, max(y_test_com[:, 0].tolist()))
        save_region_classification_map(country, algorithm, tt_split, "cnn", rep, max(y_test_com[:, 0].tolist()))
        plot_confusion_matrix(country, algorithm, tt_split, "cnn", rep, max(y_test_com[:, 0].tolist()))
        #plot_roc_auc(country, algorithm, "cnn", rep, max(y_test_com[:, 0].tolist()), test_probabilities)

def train_test_model(algorithm, model, data_loader, criterion, optimizer):
    if algorithm == 'classification':
        model.train()
    else:
        model.eval()
    
    running_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs, _ = model(inputs)  # Outputs should be of shape (N, C)
        targets = targets.view(-1)  # Ensure targets are of shape (N,)
        if algorithm == 'classification':
            
            adjusted_targets = targets - 1
            loss = criterion(outputs, adjusted_targets)
            
        else:
            loss = criterion(outputs.squeeze(), targets)
       
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
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

def cnn(rep, algorithm,r_split, country, tt_split):
    log(country,"Begin CNN on population and land cover data")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load data
    X_train, X_test, y_train, y_test, y_train_com, y_test_com = load_data(rep, country)

    # Reshape data
    L = conf.cnn_settings[country]['length']  # width and length of pixel patches
    #X_train, X_test = reshape_data(X_train, X_test, L)
    input_channels = conf.cnn_settings[country]['length']
    # Initialize parameters
    (
        hm_epochs,
        batch_size,
        nbfilter1,
        nbfilter2,
        nbfilter3,
        shapeconv,
        shapepool,
        finalshape,
        num_classes,
        early_stopping_patience
    ) = initialize_parameters(L)

    # Print confirmation
    log(country, "Data loading and reshaping complete.")
    log(country, f"Training data shape: {X_train.shape}")
    log(country, f"Test data shape: {X_test.shape}")
    log(country, f"Training Y shape: {y_train.shape}")
    log(country, f"Test Y shape: {y_test.shape}")
    log(country, "Parameters initialized.")
 
    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long if algorithm == 'classification' else torch.float32),
       
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long if algorithm == 'classification' else torch.float32),
        
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = ConvNet(algorithm, nbfilter1, nbfilter2, nbfilter3, shapeconv, shapepool, finalshape, L).to(device)
    
    # Initialize model, criterion, optimizer, and scheduler
    if os.path.exists(os.path.join(conf.OUTPUT_DIR, country, "models",  "cnn", tt_split,  algorithm, rep, 'cnn_epa.pth')):
        log(country, "Best CNN Model Loaded at : "+ os.path.join(conf.OUTPUT_DIR, country, "models",  "cnn", tt_split, algorithm, rep, 'cnn_epa.pth'))
        checkpoint = torch.load(os.path.join(conf.OUTPUT_DIR, country, "models",  "cnn", tt_split, algorithm, rep, 'cnn_epa.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        best_ep = checkpoint['best_ep']
        best_test_R2 = checkpoint['best_test_R2']
   
    criterion = nn.CrossEntropyLoss() if algorithm == 'classification' else nn.MSELoss()
    
    # Define loss function and optimizer
    learning_rate = 0.001
    #learning_rate = 0.01 if algorithm == 'classification' else 0.001
    # if algorithm == "classification":
    #     optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience= early_stopping_patience, factor=0.5)
    # Training loop
    
    best_test_R2 = -float("inf")
    best_test_loss = float("inf")
    start_index = 0

    # Start timing
    start_time = time.time()
    for epoch in range(hm_epochs):
        model.train()
        epoch_loss = 0        
        # Train the model  (algorithm, model, num_batches, batch_size, X_tensor, y_tensor, criterion, optimizer)
        train_loss, train_score, train_targets, train_predictions, train_probabilities = train_test_model(algorithm, model,  train_loader,criterion, optimizer)
        
        # Evaluate the model
        test_loss, test_score, test_targets, test_predictions, test_probabilities = train_test_model(algorithm, model, test_loader, criterion, optimizer)

        # Logging and saving best model
        log(country, f"Epoch {epoch+1}/{hm_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Train Score: {train_score:.6f}, Test Score: {test_score:.6f}")
        
        if (algorithm == 'classification' and test_score > best_test_R2) or (algorithm != 'classification' and test_score > best_test_R2 and test_loss < best_test_loss):
            best_test_loss = test_loss
            best_test_R2 = test_score
            best_epoch = epoch + 1
            save_model(country, rep, tt_split, algorithm, model, best_test_loss, best_test_R2, best_epoch)
            extract_and_save_cnn_features(model, torch.tensor(train_targets, dtype=torch.float32), algorithm, country, rep, tt_split, "train")
            extract_and_save_cnn_features(model, torch.tensor(train_predictions, dtype=torch.float32), algorithm, country, rep, tt_split, "test")
            save_results(country, algorithm, rep, tt_split, test_targets, test_predictions, y_test_com, test_probabilities)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if (epochs_no_improve >= early_stopping_patience) and best_test_R2 > 0:
            log(country, f'Early stopping at epoch {epoch+1}')
            break
        scheduler.step(test_loss)
               
    #Utilize the best model
    log(country, "Best CNN Model Saved at : "+ os.path.join(conf.OUTPUT_DIR, country, "models",  "cnn", tt_split,  algorithm, rep, 'cnn_epa.pth'))
    checkpoint = torch.load(os.path.join(conf.OUTPUT_DIR, country, "models",  "cnn", tt_split,  algorithm, rep, 'cnn_epa.pth'))
    best_ep = checkpoint['best_ep']
    best_test_R2 = checkpoint['best_test_R2']
    log(country, f"Test R2 associate with best Score: {best_test_R2:.6f}  reached at epoch: {best_ep}")

    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    log(country, f"Total training time: {total_time:.2f} seconds")
    log(country, "End CNN on population and land cover data")
        
        