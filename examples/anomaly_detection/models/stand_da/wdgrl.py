import torch
import torch.nn as nn
from typing import List
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """Feature extractor network."""
        super().__init__()
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(hidden_dims) - 1:  # only add ReLU for intermediate layers
                layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """Domain critic network."""
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WDGRL:
    def __init__(
        self,
        input_dim: int = 2,
        generator_hidden_dims: List[int] = [32, 16, 8, 4, 2],
        critic_hidden_dims: List[int] = [32, 16, 8, 4, 2],
        gamma: float = 0.1,
        _lr_generator: float = 1e-2,
        _lr_critic: float = 1e-2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.gamma = gamma
        self.device = device
        self.generator = Generator(input_dim, generator_hidden_dims).to(self.device)
        self.critic = Critic(generator_hidden_dims[-1], critic_hidden_dims).to(
            self.device
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=_lr_generator
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=_lr_critic
        )

    def compute_gradient_penalty(
        self, source_data: torch.Tensor, target_data: torch.Tensor
    ) -> torch.Tensor:
        alpha = torch.rand(source_data.size(0), 1).to(self.device)
        differences = target_data - source_data
        interpolates = source_data + (alpha * differences)
        interpolates = torch.stack(
            [interpolates, source_data, target_data]
        ).requires_grad_()

        preds = self.critic(interpolates)
        gradients = torch.autograd.grad(
            preds,
            interpolates,
            grad_outputs=torch.ones_like(preds),
            retain_graph=True,
            create_graph=True,
        )[0]
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty

    def train(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        num_epochs: int = 100,
        dc_iter: int = 100,
    ) -> List[float]:
        self.generator.train()
        self.critic.train()
        losses = []
        source_critic_scores = []
        target_critic_scores = []
        for epoch in num_epochs:
            loss = 0
            total_loss = 0
            for (source_data, _), (target_data, _) in zip(source_loader, target_loader):
                source_data, target_data = (
                    source_data.to(self.device),
                    target_data.to(self.device),
                )

                # Train domain critic
                for _ in range(dc_iter):
                    self.critic_optimizer.zero_grad()

                    with torch.no_grad():
                        source_features = self.generator(source_data).view(
                            source_data.size(0), -1
                        )
                        target_features = self.generator(target_data).view(
                            target_data.size(0), -1
                        )

                    # Compute empirical Wasserstein distance
                    dc_source = self.critic(source_features)
                    dc_target = self.critic(target_features)
                    wasserstein_distance = dc_source.mean() - dc_target.mean()

                    # Gradient penalty
                    gradient_penalty = self.compute_gradient_penalty(
                        source_features, target_features
                    )

                    # Domain critic loss
                    dc_loss = -wasserstein_distance + self.gamma * gradient_penalty
                    # print(
                    #     f"- iteration #{_} / {dc_iter} | source critic: {dc_source.mean().item()} | target critic: {dc_target.mean().item()} | wasserstein distance: {wasserstein_distance.item()} | gradient penalty: {gradient_penalty.item()}"
                    # )
                    dc_loss.backward()
                    self.critic_optimizer.step()
                    with torch.no_grad():
                        total_loss += wasserstein_distance.item()
                # Train feature extractor
                self.generator_optimizer.zero_grad()
                source_features = self.generator(source_data)
                target_features = self.generator(target_data)
                dc_source = self.critic(source_features)
                dc_target = self.critic(target_features)
                wasserstein_distance = dc_source.mean() - dc_target.mean()
                wasserstein_distance.backward()
                self.generator_optimizer.step()
                with torch.no_grad():
                    loss += wasserstein_distance.item()

            source_critic_scores.append(
                self.criticize(source_loader.dataset.tensors[0].to(self.device))
            )
            target_critic_scores.append(
                self.criticize(target_loader.dataset.tensors[0].to(self.device))
            )
            losses.append(loss / len(source_loader))
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss / len(source_loader)}")
            print("--------------------------------")
        return losses, source_critic_scores, target_critic_scores

    @torch.no_grad()
    def extract_feature(self, x: torch.Tensor) -> torch.Tensor:
        self.generator.eval()
        return self.generator(x)

    @torch.no_grad()
    def criticize(self, x: torch.Tensor) -> float:
        self.generator.eval()
        self.critic.eval()
        return self.critic(self.generator(x)).mean().item()
