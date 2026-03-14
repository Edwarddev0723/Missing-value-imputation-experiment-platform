"""GAIN — Generative Adversarial Imputation Nets (D3).

Yoon et al., 2018.
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


@METHOD_REGISTRY.register("gain")
class GAINImputer(BaseImputer):
    """GAIN: Generative Adversarial Imputation Nets — D3."""

    name = "gain"
    family = FAMILY_DL

    def __init__(
        self,
        generator_hidden: list[int] | None = None,
        discriminator_hidden: list[int] | None = None,
        epochs: int = 300,
        learning_rate: float = 1e-3,
        alpha: float = 100.0,
        batch_size: int = 256,
        device: str = "mps",
        early_stopping_patience: int = 10,
        hint_rate: float = 0.9,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            generator_hidden=generator_hidden,
            discriminator_hidden=discriminator_hidden,
            epochs=epochs,
            learning_rate=learning_rate,
            alpha=alpha,
            batch_size=batch_size,
            device=device,
            early_stopping_patience=early_stopping_patience,
            hint_rate=hint_rate,
            **kwargs,
        )
        self.gen_hidden = generator_hidden or [256, 128]
        self.disc_hidden = discriminator_hidden or [256, 128]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.batch_size = batch_size
        self.device_request = device
        self.early_stopping_patience = early_stopping_patience
        self.hint_rate = hint_rate
        self._generator = None
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
        mask = (~np.isnan(X_arr)).astype(np.float32)  # 1=observed, 0=missing
        self._init_values = np.nanmean(X_arr, axis=0)
        X_filled = X_arr.copy()
        for j in range(X_filled.shape[1]):
            X_filled[np.isnan(X_filled[:, j]), j] = self._init_values[j]

        dim = X_filled.shape[1]
        rng = np.random.RandomState(self._seed)

        # Generator: (x, m) -> x_imputed
        gen_layers = []
        prev_d = dim * 2
        for h in self.gen_hidden:
            gen_layers.extend([nn.Linear(prev_d, h), nn.ReLU()])
            prev_d = h
        gen_layers.append(nn.Linear(prev_d, dim))
        gen_layers.append(nn.Sigmoid())
        generator = nn.Sequential(*gen_layers).to(torch_device)

        # Discriminator: (x, h) -> probability each value is observed
        disc_layers = []
        prev_d = dim * 2
        for h in self.disc_hidden:
            disc_layers.extend([nn.Linear(prev_d, h), nn.ReLU()])
            prev_d = h
        disc_layers.append(nn.Linear(prev_d, dim))
        disc_layers.append(nn.Sigmoid())
        discriminator = nn.Sequential(*disc_layers).to(torch_device)

        G_opt = torch.optim.Adam(generator.parameters(), lr=self.learning_rate)
        D_opt = torch.optim.Adam(discriminator.parameters(), lr=self.learning_rate)

        X_tensor = torch.tensor(X_filled, dtype=torch.float32)
        M_tensor = torch.tensor(mask, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, M_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_g_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            g_loss_sum = 0.0
            n_batches = 0
            for batch_x, batch_m in loader:
                batch_x = batch_x.to(torch_device)
                batch_m = batch_m.to(torch_device)
                bs = batch_x.shape[0]

                # Random noise
                z = torch.rand(bs, dim).to(torch_device)
                # Hint vector
                hint_mask = torch.tensor(
                    rng.binomial(1, self.hint_rate, (bs, dim)),
                    dtype=torch.float32,
                ).to(torch_device)
                h = batch_m * hint_mask

                # Generator
                g_input = torch.cat([batch_x * batch_m + z * (1 - batch_m), batch_m], dim=1)
                g_out = generator(g_input)
                x_hat = batch_x * batch_m + g_out * (1 - batch_m)

                # Discriminator
                d_input = torch.cat([x_hat.detach(), h], dim=1)
                d_out = discriminator(d_input)

                # D loss: binary cross-entropy on whether position is real
                d_loss = -torch.mean(
                    batch_m * torch.log(d_out + 1e-8)
                    + (1 - batch_m) * torch.log(1 - d_out + 1e-8)
                )

                D_opt.zero_grad()
                d_loss.backward()
                D_opt.step()

                # G loss
                d_input_g = torch.cat([x_hat, h], dim=1)
                d_out_g = discriminator(d_input_g)
                g_loss_adv = -torch.mean((1 - batch_m) * torch.log(d_out_g + 1e-8))
                g_loss_mse = torch.mean((batch_m * (batch_x - g_out)) ** 2) / (batch_m.sum() + 1e-8)
                g_loss = g_loss_adv + self.alpha * g_loss_mse

                G_opt.zero_grad()
                g_loss.backward()
                G_opt.step()

                g_loss_sum += g_loss.item()
                n_batches += 1

            avg_g_loss = g_loss_sum / max(n_batches, 1)
            if avg_g_loss < best_g_loss - 1e-6:
                best_g_loss = avg_g_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.debug(f"GAIN early stopping at epoch {epoch + 1}")
                    break

        self._generator = generator
        self._device = torch_device
        self._dim = dim

    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        import torch

        if self._generator is None:
            raise RuntimeError("GAIN must be fit before impute.")

        X_arr = X_missing.values.astype(np.float32)
        mask = (~np.isnan(X_arr)).astype(np.float32)
        X_filled = X_arr.copy()
        for j in range(X_filled.shape[1]):
            X_filled[np.isnan(X_filled[:, j]), j] = self._init_values[j]

        self._generator.eval()
        with torch.no_grad():
            z = torch.rand(X_filled.shape[0], self._dim).to(self._device)
            m_tensor = torch.tensor(mask, dtype=torch.float32).to(self._device)
            x_tensor = torch.tensor(X_filled, dtype=torch.float32).to(self._device)
            g_input = torch.cat([x_tensor * m_tensor + z * (1 - m_tensor), m_tensor], dim=1)
            g_out = self._generator(g_input).cpu().numpy()

        result = X_arr.copy()
        miss_mask = np.isnan(result)
        result[miss_mask] = g_out[miss_mask]

        for j in range(result.shape[1]):
            still_nan = np.isnan(result[:, j])
            if still_nan.any():
                result[still_nan, j] = self._init_values[j]

        return pd.DataFrame(result, columns=self._columns, index=X_missing.index)
