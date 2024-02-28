'''
The main code of our paper for defining network, training, this code is partially followed by huaningliu's work at "https://github.com/huaningliu/DeepQuantile" 
and Tansey's work at https://github.com/tansey/quantile-regression. Thank you for their great effort!
'''
from distutils.command.config import config
from random import weibullvariate
from typing import OrderedDict
import numpy as np
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from utils import create_folds, batches
from torch_utils import clip_gradient, logsumexp


'''Neural network to map from X to quantile(s) of y.'''
class QuantileNetworkModule(nn.Module):
    def __init__(self, X_means, X_stds, y_mean, y_std, n_out, config):
        super(QuantileNetworkModule, self).__init__()
        self.X_means = X_means
        self.X_stds = X_stds
        self.y_mean = y_mean
        self.y_std = y_std
        self.n_in = X_means.shape[1]
        self.n_out = n_out
        self.layer_specs = config['layer_specs']
        # check if the input regularization weight is legal
        if config['L_1_weight'] != -1 and not (config['L_1_weight'] >= 0 and config['L_1_weight'] <= 1):
            raise Exception("L1 weight as" + str(config['L_1_weight']) + " is not legal")
        self.L_1_weight = config['L_1_weight']
        # specify activation function
        if config['activation'] == "ReLU":
            self.activation = nn.ReLU
        elif config['activation'] == "tanh":
            self.activation = nn.Tanh
        elif config['activation'] == "sigmoid":
            self.activation = nn.Sigmoid
        else:
            raise Exception(config['activation'] + "is currently not available as a activation function")
        # model structure
        mdl_structure = []

        out_dim = self.n_out if len(self.y_mean.shape) == 1 else self.n_out * self.y_mean.shape[1]
        layers = [self.n_in] + self.layer_specs
        for i in range(len(layers)-1):
            mdl_structure.append(nn.Linear(layers[i], layers[i+1]))
            mdl_structure.append(nn.Dropout(config['dropout_proportion']))
            mdl_structure.append(self.activation())
        mdl_structure.append(nn.Linear(layers[i+1], out_dim))
        
        

        self.fc_in = nn.Sequential(*mdl_structure)
        #self.fc_in.apply(weight_init)
        # self.fc_in = nn.Sequential(nn.Linear(X_means.shape[0], n_out))
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        fout = self.fc_in(x)

        # If we are dealing with multivariate responses, reshape to the (d x q) dimensions
        if len(self.y_mean.shape) != 1:
            fout = fout.reshape((-1, self.y_mean.shape[1], self.n_out))

        # If we only have 1 quantile, no need to do anything else
        if self.n_out == 1:
            return fout

        # Enforce monotonicity of the quantiles
        return torch.cat((fout[...,0:1], fout[...,0:1] + torch.cumsum(self.softplus(fout[...,1:]), dim=-1)), dim=-1)
        #return fout

    def compute_l1_loss(self, w):
      return torch.abs(w).sum()
        
    def predict(self, X):
        self.eval()
        self.zero_grad()
        #tX = autograd.Variable(torch.FloatTensor((X - self.X_means) / self.X_stds), requires_grad=False)
        #fout = self.forward(tX)
        tX = torch.FloatTensor(X).detach()

        fout = self.forward(tX)
        #return fout.data.numpy() * self.y_std[...,None] + self.y_mean[...,None]
        return fout.data.numpy()
    
class QuantileNetwork:
    def __init__(self, quantiles, config, loss='marginal'):
        self.quantiles = quantiles
        self.label = 'Quantile Network'
        self.filename = 'nn'
        self.lossfn = loss
        self.config = config
        if self.lossfn != 'marginal':
            self.label += f' ({self.lossfn})'

    def fit(self, X, y):
        self.model = fit_quantiles(X, y, config=self.config, quantiles=self.quantiles)

    def weighted_fit(self, X, y, X_tr, X_te, nr, true_weight, t_idx,dim):
        self.model = fit_weighted_quantiles(X, y, X_tr, X_te, nr, true_weight, t_idx,dim, config=self.config, quantiles=self.quantiles)   

    
    def predict(self, X):
        return self.model.predict(X)
    



    


def fit_quantiles(X, y,  config,  quantiles=0.5, 
                    nepochs=100, val_pct=0.1,
                    batch_size=None, target_batch_pct=0.01,
                    min_batch_size=20, max_batch_size=100,
                    verbose=False, lr=0.1, weight_decay=0.0, patience=5,
                    init_model=None, splits=None, file_checkpoints=True,
                    clip_gradients=False, **kwargs):
    if file_checkpoints:
        import uuid
        tmp_file = 'tmp/tmp_file_' + str(uuid.uuid4())

    # fix batch_size
    if batch_size is None:
        batch_size = min(X.shape[0], max(min_batch_size, min(max_batch_size, int(np.round(X.shape[0]*target_batch_pct)))))
        if verbose:
            print('Auto batch size chosen to be {}'.format(batch_size))

    # Standardize the features and response (helps with gradient propagation)
    Xmean = X.mean(axis=0, keepdims=True)
    Xstd = X.std(axis=0, keepdims=True)
    Xstd[Xstd == 0] = 1 # Handle constant features
    ymean, ystd = y.mean(axis=0, keepdims=True), y.std(axis=0, keepdims=True)
    #tX = autograd.Variable(torch.FloatTensor((X - Xmean) / Xstd), requires_grad=False)
    #tY = autograd.Variable(torch.FloatTensor((y - ymean) / ystd), requires_grad=False)
    tX = torch.FloatTensor(X).detach()
    tY = torch.FloatTensor(y).detach()
    # Create train/validate splits
    if splits is None:
        indices = np.arange(X.shape[0], dtype=int)
        np.random.shuffle(indices)
        train_cutoff = int(np.round(len(indices)*(1-val_pct)))
        train_indices = indices[:train_cutoff]
        validate_indices = indices[train_cutoff:]
    else:
        train_indices, validate_indices = splits

    if np.isscalar(quantiles):
        quantiles = np.array([quantiles])
 
    tquantiles = torch.FloatTensor(quantiles).detach()

    # Initialize the model
    model = QuantileNetworkModule(Xmean, Xstd, ymean, ystd, quantiles.shape[0], config) if init_model is None else init_model
    
    


    # Save the model to file
    if file_checkpoints:
        torch.save(model, tmp_file)
    else:
        import pickle
        model_str = pickle.dumps(model)
    
    if config['L_1_weight'] == -1:
        weight_decay = 0
    else:
        weight_decay = (1 - config['L_1_weight']) * config['lambda']

    # Setup the SGD method
    if config['optimizer'] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, nesterov=config['nesterov'], momentum=config['momentum'])
    elif config['optimizer'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise Exception(config['optimizer'] + "is currently not available")

    # the learning rate delays by 0.5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['lr_decay'])

    # Track progress
    train_losses, val_losses, best_loss = np.zeros(nepochs), np.zeros(nepochs), None
    num_bad_epochs = 0

    if verbose:
        print('ymax and min:', tY.max(), tY.min())

    # Univariate quantile loss
    def quantile_loss(yhat, tidx):
        z = tY[tidx,None] - yhat
        return torch.max(tquantiles[None]*z, (tquantiles[None] - 1)*z)

    # Marginal quantile loss for multivariate response
    def marginal_loss(yhat, tidx):
        z = tY[tidx,:,None] - yhat
        return torch.max(tquantiles[None,None]*z, (tquantiles[None,None] - 1)*z)

    # Geometric quantile loss -- uses a Euclidean unit ball definition of multivariate quantiles
    def geometric_loss(yhat, tidx):
        z = tY[tidx,:,None] - yhat
        return torch.norm(z, dim=1) + (z * tquantiles[None,None]).sum(dim=1)

    # Create the quantile loss function
    if len(tY.shape) == 1 or tY.shape[1] == 1:
        lossfn = quantile_loss
    
            

    # Train the model
    for epoch in range(nepochs):
        if verbose:
            print('\t\tEpoch {}'.format(epoch+1))
            sys.stdout.flush()

        # Track the loss curves
        train_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(train_indices, batch_size, shuffle=True)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tBatch {}'.format(batch_idx))
            tidx = torch.LongTensor(batch).detach()

            # Set the model to training mode
            model.train()

            # Reset the gradient
            model.zero_grad()

            # Run the model and get the predicted quantiles
            yhat = model(tX[tidx])
            
            # Loss for all quantiles
            loss = lossfn(yhat, tidx).mean()

            # add L1 regularization
            l1_weight = config['L_1_weight']
            if l1_weight != -1 and l1_weight != 0:
                l1_parameters = []
                for param in model.parameters():
                    l1_parameters.append(param.view(-1))
                l1 = config['lambda'] * l1_weight * model.compute_l1_loss(torch.cat(l1_parameters))
                loss += l1

            # Calculate gradients
            loss.backward()

            # Clip the gradients
            if clip_gradients:
                clip_gradient(model)

            # Apply the update
            # [p for p in model.parameters() if p.requires_grad]
            optimizer.step()

            # Track the loss
            train_loss += loss.data

            if np.isnan(loss.data.numpy()):
                import warnings
                warnings.warn('NaNs encountered in training model.')
                break

        validate_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(validate_indices, batch_size, shuffle=False)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tValidation Batch {}'.format(batch_idx))
            tidx = torch.LongTensor(batch).detach()

            # Set the model to test mode
            model.eval()

            # Reset the gradient
            model.zero_grad()

            # Run the model and get the conditional mixture weights
            yhat = model(tX[tidx])

            # Track the loss
            validate_loss += lossfn(yhat, tidx).sum()

        train_losses[epoch] = train_loss.data.numpy() / float(len(train_indices))
        val_losses[epoch] = validate_loss.data.numpy() / float(len(validate_indices))

        # Adjust the learning rate down if the validation performance is bad
        if num_bad_epochs > patience:
            if verbose:
                print('Decreasing learning rate to {}'.format(lr*0.5))
            scheduler.step(val_losses[epoch])
            lr *= 0.5
            num_bad_epochs = 0

        # If the model blew up and gave us NaNs, adjust the learning rate down and restart
        if np.isnan(val_losses[epoch]):
            if verbose:
                print('Network went to NaN. Readjusting learning rate down by 50%')
            if file_checkpoints:
                os.remove(tmp_file)
            return fit_quantiles(X, y, quantiles=quantiles, 
                    nepochs=nepochs, val_pct=val_pct,
                    batch_size=batch_size, target_batch_pct=target_batch_pct,
                    min_batch_size=min_batch_size, max_batch_size=max_batch_size,
                    verbose=verbose, lr=lr*0.5, weight_decay=weight_decay, patience=patience,
                    init_model=init_model, splits=splits, file_checkpoints=file_checkpoints,  **kwargs)

        # Check if we are currently have the best held-out log-likelihood
        if epoch == 0 or val_losses[epoch] <= best_loss:
            if verbose:
                print('\t\t\tSaving test set results.      <----- New high water mark on epoch {}'.format(epoch+1))
            # If so, use the current model on the test set
            best_loss = val_losses[epoch]
            if file_checkpoints:
                torch.save(model, tmp_file)
            else:
                import pickle
                model_str = pickle.dumps(model)
        else:
            num_bad_epochs += 1
        
        if verbose:
            print('Validation loss: {} Best: {}'.format(val_losses[epoch], best_loss))

    # Load the best model and clean up the checkpoints
    if file_checkpoints:
        model = torch.load(tmp_file)
        os.remove(tmp_file)
    else:
        import pickle
        model = pickle.loads(model_str)

    # Return the conditional density model that marginalizes out the grid
    return model

class ratio_network(nn.Module):
    def __init__(self, D_in, H, D_out,max_value):
        super(ratio_network, self).__init__()
        
        self.max_value = max_value
        self.relu_model= nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, H), 
            nn.ReLU(),    
            nn.Linear(H, H), 
            nn.ReLU(),
            nn.Linear(H,D_out)
    )     
        
        self.relu_model.apply(weight_init)


    def forward(self,x):

        output = self.relu_model(x)
        output = torch.clamp(output,max=self.max_value)
        return output
    

def ratio_estimation(X_tr, X_te, n, dim, H=100, D_out=1, learning_rate = 1e-4, n_epo=1000,truncated_value=4):
    ratio_model = ratio_network(D_in=dim, H=H, D_out=D_out,max_value=truncated_value)
    X_train=torch.from_numpy(X_tr)
    X_test=torch.from_numpy(X_te)



    learning_rate = 1e-4
    optimizer = torch.optim.Adam(ratio_model.parameters(), lr=learning_rate)

    for it in range(n_epo):
       u_train = ratio_model(X_train.float()) 
       u_test = ratio_model(X_test.float())
    
       # compute loss
       loss =  torch.mean(u_train**2)/2-torch.mean(u_test)# computation graph
        
       optimizer.zero_grad()
       # Backward pass
       loss.backward()
    
       # update model parameters
       optimizer.step()   
       ratio_model.eval()   
    
    return ratio_model

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def fit_weighted_quantiles(X, y, X_tr, X_te, nr, true_weight, t_idx, dim, config, quantiles=0.5,
                    nepochs=100, val_pct=0.1,
                    batch_size=None, target_batch_pct=0.01,
                    min_batch_size=20, max_batch_size=100,
                    verbose=False, lr=0.1, weight_decay=0.0, patience=5,
                    init_model=None, splits=None, file_checkpoints=True,
                    clip_gradients=False, **kwargs):
    if file_checkpoints:
        import uuid
        tmp_file = 'tmp/tmp_file_' + str(uuid.uuid4())

    # fix batch_size
    if batch_size is None:
        batch_size = min(X.shape[0], max(min_batch_size, min(max_batch_size, int(np.round(X.shape[0]*target_batch_pct)))))
        if verbose:
            print('Auto batch size chosen to be {}'.format(batch_size))

    # Standardize the features and response (helps with gradient propagation)
    Xmean = X.mean(axis=0, keepdims=True)
    Xstd = X.std(axis=0, keepdims=True)
    Xstd[Xstd == 0] = 1 # Handle constant features
    ymean, ystd = y.mean(axis=0, keepdims=True), y.std(axis=0, keepdims=True)

    tX = torch.FloatTensor(X).detach()
    tY = torch.FloatTensor(y).detach()


    true_weight = torch.FloatTensor(true_weight).detach()
      
    # Create train/validate splits
    if splits is None:
        indices = np.arange(X.shape[0], dtype=int)
        np.random.shuffle(indices)
        train_cutoff = int(np.round(len(indices)*(1-val_pct)))
        train_indices = indices[:train_cutoff]
        validate_indices = indices[train_cutoff:]
    else:
        train_indices, validate_indices = splits

    if np.isscalar(quantiles):
        quantiles = np.array([quantiles])
   
    tquantiles = torch.FloatTensor(quantiles).detach()

    # Initialize the model
    model = QuantileNetworkModule(Xmean, Xstd, ymean, ystd, quantiles.shape[0], config) if init_model is None else init_model

    # Save the model to file
    if file_checkpoints:
        torch.save(model, tmp_file)
    else:
        import pickle
        model_str = pickle.dumps(model)
    
    if config['L_1_weight'] == -1:
        weight_decay = 0
    else:
        weight_decay = (1 - config['L_1_weight']) * config['lambda']

    # Setup the SGD method
    if config['optimizer'] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, nesterov=config['nesterov'], momentum=config['momentum'])
    elif config['optimizer'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise Exception(config['optimizer'] + "is currently not available")

    # the learning rate delays by 0.5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['lr_decay'])

    # Track progress
    train_losses, val_losses, best_loss = np.zeros(nepochs), np.zeros(nepochs), None
    num_bad_epochs = 0

    if verbose:
        print('ymax and min:', tY.max(), tY.min())

    # Univariate quantile loss
    def weighted_quantile_loss(yhat, tidx, weighted):
        z = tY[tidx,None] - yhat
        return weighted*torch.max(tquantiles[None]*z, (tquantiles[None] - 1)*z)

    # Marginal quantile loss for multivariate response


    # Create the quantile loss function
    lossfn = weighted_quantile_loss

    if t_idx == 1:
       ratio = ratio_estimation(X_tr, X_te, nr, dim)


    # Train the model
    for epoch in range(nepochs):
        if verbose:
            print('\t\tEpoch {}'.format(epoch+1))
            sys.stdout.flush()

        # Track the loss curves
        train_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(train_indices, batch_size, shuffle=True)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tBatch {}'.format(batch_idx))
            tidx = torch.LongTensor(batch).detach()

            # Set the model to training mode
            model.train()

            # Reset the gradient
            model.zero_grad()

            # Run the model and get the predicted quantiles
            yhat = model(tX[tidx])
            if t_idx == 1:
               weight = ratio(tX[tidx].float()) 
            else:
               weight = true_weight[tidx,None] 
            
            weight = weight.detach()
            
            loss = lossfn(yhat, tidx, weight).mean()

        

            # Calculate gradients
            loss.backward()

            # Clip the gradients
            if clip_gradients:
                clip_gradient(model)

            # Apply the update
            # [p for p in model.parameters() if p.requires_grad]
            optimizer.step()

            # Track the loss
            train_loss += loss.data

            if np.isnan(loss.data.numpy()):
                import warnings
                warnings.warn('NaNs encountered in training model.')
                break

   
        validate_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(validate_indices, batch_size, shuffle=False)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tValidation Batch {}'.format(batch_idx))
            tidx = torch.LongTensor(batch).detach()

            # Set the model to test mode
            model.eval()

            # Reset the gradient
            model.zero_grad()

            # Run the model and get the conditional mixture weights
            yhat = model(tX[tidx])
            if t_idx == 1:
               weight = ratio(tX[tidx].float()) 
    
            else:
               weight = true_weight[tidx,None] 
            
            
            weight = weight.detach()
            
            
            # Track the loss
            validate_loss += lossfn(yhat, tidx, weight).sum()

        train_losses[epoch] = train_loss.data.numpy() / float(len(train_indices))
        val_losses[epoch] = validate_loss.data.numpy() / float(len(validate_indices))
        train_losses[epoch] = train_loss.data.numpy() / float(len(train_indices))
    

        # Adjust the learning rate down if the validation performance is bad
        if num_bad_epochs > patience:
            if verbose:
                print('Decreasing learning rate to {}'.format(lr*0.5))
            scheduler.step(val_losses[epoch])
            lr *= 0.5
            num_bad_epochs = 0

        # If the model blew up and gave us NaNs, adjust the learning rate down and restart
        if np.isnan(val_losses[epoch]):
            if verbose:
                print('Network went to NaN. Readjusting learning rate down by 50%')
            if file_checkpoints:
                os.remove(tmp_file)
            return  fit_weighted_quantiles(X, y, X_tr, X_te, nr, true_weight, t_idx, config, quantiles=0.5,
                    nepochs=100, val_pct=0.1,
                    batch_size=None, target_batch_pct=0.01,
                    min_batch_size=20, max_batch_size=100,
                    verbose=False, lr=0.1, weight_decay=0.0, patience=5,
                    init_model=None, splits=None, file_checkpoints=True,
                    clip_gradients=False, **kwargs)

        # Check if we are currently have the best held-out log-likelihood
        if epoch == 0 or val_losses[epoch] <= best_loss:
            if verbose:
                print('\t\t\tSaving test set results.      <----- New high water mark on epoch {}'.format(epoch+1))
            # If so, use the current model on the test set
            best_loss = val_losses[epoch]
            if file_checkpoints:
                torch.save(model, tmp_file)
            else:
                import pickle
                model_str = pickle.dumps(model)
        else:
            num_bad_epochs += 1
        
        if verbose:
            print('Validation loss: {} Best: {}'.format(val_losses[epoch], best_loss))

    # Load the best model and clean up the checkpoints
    if file_checkpoints:
        model = torch.load(tmp_file)
        os.remove(tmp_file)
    else:
        import pickle
        model = pickle.loads(model_str)

    # Return the conditional density model that marginalizes out the grid
    return model



