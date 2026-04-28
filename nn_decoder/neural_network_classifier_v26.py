# -*- coding: utf-8 -*-
"""
Neural Network Classifier and Loss Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import torch.nn.init as init

# ==========================================
# 1. Divergence and Loss Metrics
# ==========================================

def cross_entropy(X, Y):
    """ X is Prediction, Y is Target """
    return -torch.sum(Y * torch.log(X + torch.finfo(torch.float32).eps), dim=-1)

def entropy_calc(Y):
    """ Calculates entropy of a distribution Y """
    return -torch.sum(Y * torch.log(Y + torch.finfo(torch.float32).eps), dim=-1)

def KL_calc(X, Y):
    """
    Forward KL Divergence: D_KL(Target || Prediction)
    X: Prediction, Y: Target
    """
    KL = cross_entropy(X, Y) - entropy_calc(Y)
    return torch.clamp(KL, min=0.0)

def JS_calc(X, Y):
    """
    Jensen-Shannon Divergence
    X: Prediction, Y: Target
    """
    M = 0.5 * (X + Y)
    # D_KL(Target || M) -> KL_calc(M, Target)
    kl_xm = KL_calc(M, X) 
    kl_ym = KL_calc(M, Y)
    return 0.5 * kl_xm + 0.5 * kl_ym

def Wasserstein_calc_1D(X, Y):
    """
    1D Wasserstein Distance (Earth Mover's Distance)
    Calculated as the L1 distance between the Cumulative Distribution Functions (CDFs).
    """
    cdf_X = torch.cumsum(X, dim=-1)
    cdf_Y = torch.cumsum(Y, dim=-1)
    w_dist = torch.sum(torch.abs(cdf_X - cdf_Y), dim=-1)
    return w_dist

# ==========================================
# 2. Neural Network Architectures
# ==========================================

class NN_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN_classifier, self).__init__()
        # Standard configuration as per legacy architecture
        if isinstance(hidden_size, list):
            self.fc1 = nn.Linear(input_size, hidden_size[0])
            self.fc2 = nn.Linear(hidden_size[0], output_size)
        else:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class SimpleFlexibleNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        """
        Initializes a neural network with flexible hidden layers and activations.
        """
        super(SimpleFlexibleNNClassifier, self).__init__()
        
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = activations.get(activation.lower(), nn.ReLU()) 
        
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        
        layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        for layer in self.layers:
            init.xavier_uniform_(layer.weight)
            init.constant_(layer.bias, 0)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x) 
        return x

# ==========================================
# 3. Model Forward and Evaluation Logic
# ==========================================

def get_model_probabilities(model, batch_inputs, model_type):
    if model_type == 'ppc':
        integrated_inputs = torch.mean(batch_inputs, dim=0, keepdim=True)
        logits = model(integrated_inputs)
        probs = F.softmax(logits, dim=-1)
    elif model_type == 'sampling':
        logits = model(batch_inputs) 
        probs = F.softmax(logits, dim=-1)
    return probs

def custom_loss_all_H(pred_probs, targets, entropy_lambda, model_type, pcs=None, explained_variance=None, loss_func_type='JS'):
    
    # 1. Route the Logic based on Architecture
    if model_type == 'sampling':
        # Calculate Instantaneous Entropy BEFORE averaging
        instantaneous_entropy = torch.mean(entropy_calc(pred_probs)) 
        
        # Average the predictions for the base divergence loss
        pred_probs_loss = torch.mean(pred_probs, dim=0, keepdim=True)
        
        # Apply the sharpness penalty
        penalty = entropy_lambda * instantaneous_entropy
        entropy_log_val = instantaneous_entropy.item()
        
    else: # model_type == 'ppc'
        pred_probs_loss = pred_probs
        
        # DO NOT penalize the Spatial model's entropy. 
        penalty = 0.0 
        
        # Calculate for logging purposes
        entropy_log_val = torch.mean(entropy_calc(pred_probs_loss)).item()

    targets_mean = torch.mean(targets, dim=0, keepdim=True)

    # 2. Calculate Base Divergence 
    if loss_func_type == 'JS':
        loss_val = JS_calc(pred_probs_loss, targets_mean)
    elif loss_func_type == 'KL':
        loss_val = KL_calc(pred_probs_loss, targets_mean)
    elif loss_func_type == 'Wasserstein':
        loss_val = Wasserstein_calc_1D(pred_probs_loss, targets_mean)
    elif loss_func_type == 'PCA' and pcs is not None:
        pred_proj = torch.matmul(pred_probs_loss, pcs.T)
        target_proj = torch.matmul(targets_mean, pcs.T)
        loss_val = torch.sum(explained_variance * (pred_proj - target_proj)**2, dim=-1) * 100
    elif loss_func_type == 'MSE':
        # Mean Squared Error — suitable for low-dimensional soft targets (e.g. 2D decision posterior)
        loss_val = torch.mean((pred_probs_loss - targets_mean)**2, dim=-1)
    else:
        loss_val = cross_entropy(pred_probs_loss, targets_mean)

    mean_loss = torch.mean(loss_val)
    
    # 3. Total Loss: Base Divergence + (Conditional) Sharpness Penalty
    total_loss = mean_loss + penalty

    return total_loss, entropy_log_val

def evaluate_model_entropy(batch_inputs, batch_targets, model, loss_func_type, entropy_lambda, model_type, pcs, explained_variance, angles, circle_type, device):
    model.eval()
    
    with torch.no_grad():
        pred_probs = get_model_probabilities(model, batch_inputs, model_type)
        
        loss, entropy_batch = custom_loss_all_H(
            pred_probs, batch_targets, entropy_lambda, model_type, pcs, explained_variance, loss_func_type
        )

        if model_type == 'sampling':
            # This captures the instantaneous samples for your heatmaps! Shape: (1, n_angles, n_bins)
            pred_samp = np.expand_dims(pred_probs.cpu().numpy().transpose(1,0), axis=0)
            pred_m = torch.mean(pred_probs, dim=0).reshape(1,-1).cpu().numpy()
        else:
            # PPC has no instantaneous samples, returning zeros of matching shape
            pred_samp = np.zeros((1, batch_targets.shape[1], batch_inputs.shape[0]))
            pred_m = pred_probs.reshape(1,-1).cpu().numpy()
            
        targ_m = torch.mean(batch_targets, dim=0).reshape(1,-1).cpu().numpy()
        cv_val = np.zeros(1) 
        
    return loss, pred_samp, pred_m, targ_m, cv_val

# ==========================================
# 4. Training Loop
# ==========================================

def train_and_select_best_model(REP, model_type, train_loader, model_params, training_params, verbose=True):
    input_size = model_params['input_size']
    output_size = model_params['output_size']
    hidden_sizes = model_params['hidden_sizes']
    
    # Safely extract activation, default to 'relu' if not provided
    activation = model_params.get('activation_function', 'relu')
    device = training_params['device']
    minibatch_size = training_params.get('minibatch_size', 32) 
    
    best_overall_loss = float('inf')
    best_overall_model = None

    for r in range(REP):
        # Instantiate using the flexible architecture!
        model = SimpleFlexibleNNClassifier(
            input_size=input_size, 
            hidden_sizes=hidden_sizes, 
            output_size=output_size, 
            activation=activation
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'], weight_decay=1e-4)
        
        for epoch in range(training_params['num_epochs']):
            model.train()
            optimizer.zero_grad()
            count = 0
            
            for batch_inputs, batch_targets in train_loader:
                pred_probs = get_model_probabilities(model, batch_inputs, model_type)
                
                loss, _ = custom_loss_all_H(
                    pred_probs, 
                    batch_targets, 
                    training_params['entropy_lambda'], 
                    model_type,
                    training_params['pcs'], 
                    training_params['explained_variance'],
                    training_params['loss_func']
                )
                
                # Normalize loss for accumulation to maintain steady gradients
                loss = loss / minibatch_size
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                count += 1
                
                # Step optimizer only when minibatch is full
                if count % minibatch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
            # Ensure final step is taken if trailing trials don't fit perfectly in minibatch
            if count % minibatch_size != 0:
                optimizer.step()
                optimizer.zero_grad()

        # Evaluate at the end of training for this rep
        model.eval()
        rep_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in train_loader:
                pred_probs = get_model_probabilities(model, batch_inputs, model_type)
                loss, _ = custom_loss_all_H(
                    pred_probs, 
                    batch_targets, 
                    training_params['entropy_lambda'],
                    model_type,
                    training_params['pcs'], 
                    training_params['explained_variance'],
                    training_params['loss_func']
                )
                rep_loss += loss.item()

        if rep_loss < best_overall_loss:
            best_overall_loss = rep_loss
            best_overall_model = copy.deepcopy(model)
            
        if verbose:
            print(f"    Rep {r+1}/{REP} | Loss: {rep_loss:.4f} | Best: {best_overall_loss:.4f}")

    if verbose:
        print(f"  -> Best {model_type.upper()} Loss: {best_overall_loss:.4f}\n")
        
    return best_overall_model, best_overall_loss