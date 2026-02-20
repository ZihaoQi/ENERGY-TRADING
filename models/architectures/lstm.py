from typing import Optional, Type
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from base_classes import BaseModel


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class LSTMModel(BaseModel):

    def __init__(
        self,
        config: dict,
        criterion: Optional[nn.Module] = None,
        optimizer_class: Optional[Type[optim.Optimizer]] = None,
        optimizer_params=None
    ):
        super().__init__(config)

        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda',
        else:
            device = 'cpu'

        self.device = torch.device(device)

        self.model = LSTMClassifier(
            input_size=config["input_size"],
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 1),
            num_classes=config["num_classes"]
        ).to(self.device)

        # Loss
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()

        # Optimizer
        if optimizer_class:
            self.optimizer = optimizer_class(
                self.model.parameters(),
                **(optimizer_params or {})
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.get("learning_rate", 0.001)
            )

        self.epochs = config.get("epochs", 20)
        self.batch_size = config.get("batch_size", 32)

    def fit(self, X, y):

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0

            for xb, yb in loader:
                self.optimizer.zero_grad()
                outputs = self.model(xb)
                loss = self.criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}")

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = torch.argmax(outputs, dim=1)

        return preds.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()
