import torch
import torch.nn as nn
from typing import List
from torch.utils.data import DataLoader


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int],
        lr: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Feature extractor network."""
        super().__init__()
        self.device = device
        layers = []
        prev_dim = input_dim

        for hidden_dim in encoder_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        prev_dim = encoder_hidden_dims[-1]

        for hidden_dim in decoder_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, input_dim))

        self.net = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x)

    def criterion(self, outputs, inputs):
        return torch.mean(torch.abs(outputs - inputs))

    @torch.no_grad()
    def reconstruction_loss(self, x):
        losses = []
        for i in range(x.shape[0]):
            losses.append(
                self.criterion(self.forward(x[i].unsqueeze(0)), x[i].unsqueeze(0))
            )
        return losses

    def train(self, data_loader: DataLoader, num_epochs: int = 100):
        losses = []
        for epoch in range(num_epochs):
            for batch in data_loader:
                inputs = batch[0]
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, inputs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss / len(data_loader)}")
            losses.append(loss.item() / len(data_loader))
        return losses
