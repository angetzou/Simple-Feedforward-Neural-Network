import torch
import torch.nn as nn

class Dataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, item):
        return {
            "x": torch.tensor(self.features[item, :], dtype=torch.float),
            "y": torch.tensor(self.targets[item], dtype=torch.long),

        }

class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    def loss_fn(self, targets, outputs):
        return nn.CrossEntropyLoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        training_correct = 0
        num_samples = 0
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs = data["x"].to(self.device)
            targets = data["y"].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            num_samples += len(targets)
            training_correct += torch.sum(targets == torch.max(outputs, 1)[1])
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        training_accuracy = training_correct / num_samples
        return final_loss/len(data_loader), training_accuracy

    def evaluate(self, data_loader):
        self.model.eval()
        validation_correct = 0
        num_samples = 0
        final_loss = 0
        y_pred = []
        y_true = []
        for data in data_loader:
            inputs = data["x"].to(self.device)
            targets = data["y"].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            num_samples += len(targets)
            out = torch.max(outputs, 1)[1].cpu().numpy()
            y_pred.extend(out)
            aux_labels = targets.cpu().numpy()
            y_true.extend(aux_labels)
            validation_correct += torch.sum(targets == torch.max(outputs, 1)[1])
            final_loss += loss.item()
        validation_accuracy = validation_correct / num_samples
        return final_loss/len(data_loader), validation_accuracy, y_pred, y_true

class model(nn.Module):
    def __init__(self, n_features, n_targets, n_layers , hidden_size, dropout):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            if(len(layers)==0):
                layers.append(nn.Linear(n_features, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, n_targets))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


