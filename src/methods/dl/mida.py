"""MIDA — Missing Data Imputation Denoising Autoencoder (D1).

Gondara & Wang, 2018.
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


@METHOD_REGISTRY.register("mida")
class MIDAImputer(BaseImputer):
    """MIDA: Denoising Autoencoder for missing data imputation — D1."""

    name = "mida"
    family = FAMILY_DL

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        corruption_rate: float = 0.5,
        batch_size: int = 256,
        device: str = "mps",
        early_stopping_patience: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            hidden_layers=hidden_layers,
            epochs=epochs,
            learning_rate=learning_rate,
            corruption_rate=corruption_rate,
            batch_size=batch_size,
            device=device,
            early_stopping_patience=early_stopping_patience,
            **kwargs,
        )
        self.hidden_layers = hidden_layers or [256, 128, 256]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.corruption_rate = corruption_rate
        self.batch_size = batch_size
        self.device_request = device
        self.early_stopping_patience = early_stopping_patience
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
        mask = np.isnan(X_arr)
        self._init_values = np.nanmean(X_arr, axis=0)
        # Fill NaN for training
        X_filled = X_arr.copy()
        for j in range(X_filled.shape[1]):
            X_filled[mask[:, j], j] = self._init_values[j]

        input_dim = X_filled.shape[1]
        rng = np.random.RandomState(self._seed)

        # Build denoising autoencoder
        layers = []
        prev_dim = input_dim
        for h in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, input_dim))
        model = nn.Sequential(*layers).to(torch_device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        X_tensor = torch.tensor(X_filled, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        patience_counter = 0

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for (batch_x,) in loader:
                batch_x = batch_x.to(torch_device)
                # Corrupt input
                corruption_mask = torch.tensor(
                    rng.binomial(1, self.corruption_rate, size=batch_x.shape),
                    dtype=torch.float32,
                ).to(torch_device)
                corrupted = batch_x * (1 - corruption_mask)

                output = model(corrupted)
                loss = loss_fn(output, batch_x)

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
                    logger.debug(f"MIDA early stopping at epoch {epoch + 1}")
                    break

        self._model = model
        self._device = torch_device

    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        import torch

        if self._model is None:
            raise RuntimeError("MIDA must be fit before impute.")

        X_arr = X_missing.values.astype(np.float32)
        mask = np.isnan(X_arr)
        X_filled = X_arr.copy()
        for j in range(X_filled.shape[1]):
            X_filled[mask[:, j], j] = self._init_values[j]

        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_filled, dtype=torch.float32).to(self._device)
            X_imputed = self._model(X_tensor).cpu().numpy()

        # Only replace missing values
        result = X_arr.copy()
        result[mask] = X_imputed[mask]
        # Fill any remaining NaN
        for j in range(result.shape[1]):
            still_nan = np.isnan(result[:, j])
            if still_nan.any():
                result[still_nan, j] = self._init_values[j]

        return pd.DataFrame(result, columns=self._columns, index=X_missing.index)
