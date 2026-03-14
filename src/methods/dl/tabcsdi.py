"""TabCSDI — Conditional Score-based Diffusion Models for Tabular data (D4).

Zheng & Charoenphakdee, 2022.
Simplified implementation of diffusion-based imputation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.constants import FAMILY_DL
from src.methods.base import BaseImputer
from src.registry import METHOD_REGISTRY
from src.utils.device import get_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


@METHOD_REGISTRY.register("tabcsdi")
class TabCSDIImputer(BaseImputer):
    """TabCSDI: Conditional Score-based Diffusion for tabular imputation — D4."""

    name = "tabcsdi"
    family = FAMILY_DL

    def __init__(
        self,
        diffusion_steps: int = 50,
        hidden_dim: int = 128,
        n_layers: int = 4,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        device: str = "mps",
        early_stopping_patience: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            diffusion_steps=diffusion_steps,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device,
            early_stopping_patience=early_stopping_patience,
            **kwargs,
        )
        self.diffusion_steps = diffusion_steps
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device_request = device
        self.early_stopping_patience = early_stopping_patience
        self.beta_start = beta_start
        self.beta_end = beta_end
        self._model = None
        self._columns: list[str] = []
        self._init_values: np.ndarray | None = None

    def fit(self, X_train: pd.DataFrame, metadata: dict[str, Any] | None = None) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._columns = list(X_train.columns)
        device = get_device(self.device_request)
        torch_device = torch.device(device)

        X_arr = X_train.values.astype(np.float32)
        obs_mask = (~np.isnan(X_arr)).astype(np.float32)
        self._init_values = np.nanmean(X_arr, axis=0)
        X_filled = X_arr.copy()
        for j in range(X_filled.shape[1]):
            X_filled[np.isnan(X_filled[:, j]), j] = self._init_values[j]

        input_dim = X_filled.shape[1]

        # Noise schedule
        betas = np.linspace(self.beta_start, self.beta_end, self.diffusion_steps).astype(np.float32)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        self._betas = torch.tensor(betas).to(torch_device)
        self._alpha_bars = torch.tensor(alpha_bars).to(torch_device)

        # Score network: predicts noise given (x_noisy, t, mask, x_cond)
        # input: x_noisy(dim) + mask(dim) + x_cond(dim) + t_emb(hidden_dim)
        t_embed = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        layers = [nn.Linear(input_dim * 3 + self.hidden_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.n_layers - 1):
            layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(self.hidden_dim, input_dim))
        score_net = nn.Sequential(*layers)

        model = nn.ModuleDict({"t_embed": t_embed, "score_net": score_net}).to(torch_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        X_tensor = torch.tensor(X_filled, dtype=torch.float32)
        M_tensor = torch.tensor(obs_mask, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, M_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        patience_counter = 0

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_m in loader:
                batch_x = batch_x.to(torch_device)
                batch_m = batch_m.to(torch_device)
                bs = batch_x.shape[0]

                # Sample timestep
                t = torch.randint(0, self.diffusion_steps, (bs,), device=torch_device)
                t_norm = t.float().unsqueeze(1) / self.diffusion_steps

                # Forward diffusion on missing values
                alpha_bar_t = self._alpha_bars[t].unsqueeze(1)
                noise = torch.randn_like(batch_x)
                x_noisy = torch.sqrt(alpha_bar_t) * batch_x + torch.sqrt(1 - alpha_bar_t) * noise

                # Condition on observed values
                x_cond = batch_x * batch_m

                # Predict noise
                t_emb = model["t_embed"](t_norm)
                score_input = torch.cat([x_noisy, batch_m, x_cond, t_emb], dim=1)
                noise_pred = model["score_net"](score_input)

                # Loss only on missing positions
                loss_mask = 1 - batch_m
                if loss_mask.sum() > 0:
                    loss = ((noise_pred - noise) ** 2 * loss_mask).sum() / (loss_mask.sum() + 1e-8)
                else:
                    loss = ((noise_pred - noise) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.debug(f"TabCSDI early stopping at epoch {epoch + 1}")
                    break

        self._model = model
        self._device = torch_device
        self._input_dim = input_dim

    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        import torch

        if self._model is None:
            raise RuntimeError("TabCSDI must be fit before impute.")

        X_arr = X_missing.values.astype(np.float32)
        obs_mask = (~np.isnan(X_arr)).astype(np.float32)
        X_filled = X_arr.copy()
        for j in range(X_filled.shape[1]):
            X_filled[np.isnan(X_filled[:, j]), j] = self._init_values[j]

        self._model.eval()
        with torch.no_grad():
            m_tensor = torch.tensor(obs_mask, dtype=torch.float32).to(self._device)
            x_cond = torch.tensor(X_filled, dtype=torch.float32).to(self._device) * m_tensor

            # Start from noise
            x_t = torch.randn(X_arr.shape[0], self._input_dim).to(self._device)

            # Reverse diffusion
            for t in reversed(range(self.diffusion_steps)):
                t_tensor = torch.full((X_arr.shape[0], 1), t / self.diffusion_steps,
                                     dtype=torch.float32, device=self._device)
                t_emb = self._model["t_embed"](t_tensor)
                score_input = torch.cat([x_t, m_tensor, x_cond, t_emb], dim=1)
                noise_pred = self._model["score_net"](score_input)

                beta_t = self._betas[t]
                alpha_t = 1 - beta_t
                alpha_bar_t = self._alpha_bars[t]

                # DDPM update
                x_t = (1 / torch.sqrt(alpha_t)) * (
                    x_t - (beta_t / torch.sqrt(1 - alpha_bar_t + 1e-8)) * noise_pred
                )
                if t > 0:
                    x_t = x_t + torch.sqrt(beta_t) * torch.randn_like(x_t)

            # Replace observed values back
            x_result = x_cond + x_t * (1 - m_tensor)
            result = x_result.cpu().numpy()

        # Ensure no NaN
        nan_mask = np.isnan(result)
        if nan_mask.any():
            for j in range(result.shape[1]):
                result[nan_mask[:, j], j] = self._init_values[j]

        return pd.DataFrame(result, columns=self._columns, index=X_missing.index)
