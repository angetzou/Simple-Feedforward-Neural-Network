import torch
import torch.nn as nn
import optuna
import pandas as pd
import numpy as np
import utils
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from seaborn import set
from seaborn import set_style
from sklearn.preprocessing import MinMaxScaler



DEVICE = "cuda"
EPOCHS = 1000

def run_training(params ,mode, save_model = False):

    if(torch.cuda.is_available()):
        print('yes')
    else:
        print('no')

    X_train = '...Load training dataset'
    y_train = '...Load the training labels'

    # validation dataset

    x_val = '..Load validation data'
    y_val = '..Load validation labels'


    x_train = x_train.to_numpy()
    #y_train = y_train.to_numpy()
    y_train = np.array(y_train)

    x_val = x_val.to_numpy()
    #y_val = y_val.to_numpy()
    y_val = np.array(y_val)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    train_dataset = utils.Dataset(features=x_train, targets=y_train)
    valid_dataset = utils.Dataset(features=x_val, targets=y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=512, shuffle=True)

    model = utils.model_neural_network(n_features=x_train.shape[1], n_targets='set the number of targets', n_layers=params["num_layers"], hidden_size=params["hidden_size"], dropout=params["dropout"])
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])


    eng = utils.Engine(model, optimizer, device=DEVICE)

    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = eng.train(train_loader)
        valid_loss, val_acc, y_pred, y_true = eng.evaluate(valid_loader)
        print(f'{epoch}, {train_loss}, {train_acc}, {valid_loss}, {val_acc}')
        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                print('Here we should have saved the model')
                path = r'C:\Users\...' + "\\" + str(params["num_layers"]) + "_" + str(params["hidden_size"]) + "_" + \
                       str(params["dropout"]) + "_" + str(params["learning_rate"]) + "_" + str('.pth')
                torch.save(model.state_dict(), path)
        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            break
    return best_loss



DEVICE = "cuda"
EPOCHS = 1000

def run_testing(params , mode, save_model = True):

    if(torch.cuda.is_available()):
        print('yes')
    else:
        print('no')
    #loading the training and validation data for the new training
    x_train = 'Load training data'
    y_train = 'Load training data'
    x_val = 'Load validation data'
    y_val = 'Load validation data'

    x_train = np.vstack(x_train,x_val)
    y_train = np.vstack(y_train, y_val)


    x_test = 'Load test set'
    y_test = 'Load test labels'


    x_train = x_train.to_numpy()
    #y_train = y_train.to_numpy()
    y_train = np.array(y_train)

    x_test = x_test.to_numpy()
    #y_val = y_val.to_numpy()
    y_test = np.array(y_test)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    train_dataset = utils.Dataset(features=x_train, targets=y_train)
    valid_dataset = utils.Dataset(features=x_test, targets=y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=512, shuffle=True)

    model = utils.model(n_features=x_train.shape[1], n_targets='set the number of targets', n_layers=params["num_layers"], hidden_size=params["hidden_size"], dropout=params["dropout"])

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])


    eng = utils.Engine(model, optimizer, device=DEVICE)

    best_loss = np.inf
    early_stopping_iter = 10 # 20
    early_stopping_counter = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = eng.train(train_loader)
        valid_loss, val_acc, y_pred, y_true = eng.evaluate(valid_loader)
        print(f'{epoch}, {train_loss}, {train_acc}, {valid_loss}, {val_acc}')
        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                print('Here we should have saved the model')
                path = r'C:\...' + "\\" + str(params["num_layers"]) + "_" + str(params["hidden_size"]) + "_" + str(params["dropout"]) + "_" + str(params["learning_rate"]) + "_" + str(mode) + "_" + str('.pth')
                torch.save(model.state_dict(), path)
                set(font_scale=1.0)
                set_style("darkgrid")
                cf_matrix = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(40, 40))
                g = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
                fig = g.figure
                fig.savefig(r'C:\...' + "\\" + str(mode) + '.png')
        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            break
    return best_loss



def objective(trial):
    params = {
        'num_layers': trial.suggest_int("num_layers", 2, 7),
        'hidden_size': trial.suggest_int("hidden_size", 300, 2048),
        "dropout" : trial.suggest_uniform("dropout", 0.1, 0.7),
        "learning_rate" : trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)
    }
    temp_loss = run_training(params, save_model=False)
    return temp_loss



if __name__ == "__main__":

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print('best trial:')
    trial_ = study.best_trial

    print(trial_.values)
    print(trial_.params)
    # trial_.params are the best params
    score = run_testing(trial_.params, save_model=True)

    print(score)