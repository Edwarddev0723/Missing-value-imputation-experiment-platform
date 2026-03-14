"""MIRACLE — Missing data Imputation Refinement And Causal LEarning (D5).

Kyono et al., 2021.
Simplified implementation with causal regularization.
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


@METHOD_REGISTRY.register("miracle")
class MIRACLEImputer(BaseImputer):
    """MIRACLE: Causal learning-based imputation — D5."""

    name = "miracle"
    family = FAMILY_DL

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 3,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        lambda_reg: float = 1.0,
        batch_size: int = 256,
        device: str = "mps",
        early_stopping_patience: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            epochs=epochs,
            learning_rate=learning_rate,
            lambda_reg=lambda_reg,
            batch_size=batch_size,
            device=device,
            early_stopping_patience=early_stopping_patience,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.device_request = device
        self.early_stopping_patience = early_stopping_patience
        self._model = None
        self._adj_matrix = None
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

        # Imputation network
        layers = [nn.Linear(input_dim * 2, self.hidden_dim), nn.ReLU()]
        for _ in range(self.n_layers - 1):
            layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(self.hidden_dim, input_dim))
        imputer_net = nn.Sequential(*layers).to(torch_device)

        # Learnable adjacency matrix for causal structure
        adj_raw = nn.Parameter(torch.zeros(input_dim, input_dim, device=torch_device))

        optimizer = torch.optim.Adam(
            list(imputer_net.parameters()) + [adj_raw], lr=self.learning_rate
        )

        X_tensor = torch.tensor(X_filled, dtype=torch.float32)
        M_tensor = torch.tensor(obs_mask, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, M_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_m in loader:
                batch_x = batch_x.to(torch_device)
                batch_m = batch_m.to(torch_device)

                # Impute
                net_input = torch.cat([batch_x * batch_m, batch_m], dim=1)
                x_imputed = imputer_net(net_input)

                # Reconstruction loss (on observed values)
                recon_loss = ((x_imputed - batch_x) ** 2 * batch_m).sum() / (batch_m.sum() + 1e-8)

                # Causal regularization: DAG constraint (acyclicity)
                # A = sigmoid(adj_raw) with zero diagonal
                A = torch.sigmoid(adj_raw) * (1 - torch.eye(input_dim, device=torch_device))

                # h(A) = tr(e^A) - d  (NOTEARS acyclicity constraint)
                # Use matrix exponential approximation
                M_matrix = torch.eye(input_dim, device=torch_device) + A / input_dim
                M_pow = M_matrix
                for _ in range(input_dim - 1):
                    M_pow = M_pow @ M_matrix
                dag_loss = torch.trace(M_pow) - input_dim

                # Sparsity regularization
                sparsity_loss = A.abs().sum()

                loss = recon_loss + self.lambda_reg * (dag_loss + 0.01 * sparsity_loss)

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
                    logger.debug(f"MIRACLE early stopping at epoch {epoch + 1}")
                    break

        self._model = imputer_net
        self._adj_matrix = torch.sigmoid(adj_raw).detach()
        self._device = torch_device

    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        import torch

        if self._model is None:
            raise RuntimeError("MIRACLE must be fit before impute.")

        X_arr = X_missing.values.astype(np.float32)
        obs_mask = (~np.isnan(X_arr)).astype(np.float32)
        X_filled = X_arr.copy()
        for j in range(X_filled.shape[1]):
            X_filled[np.isnan(X_filled[:, j]), j] = self._init_values[j]

        self._model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_filled, dtype=torch.float32).to(self._device)
            m_tensor = torch.tensor(obs_mask, dtype=torch.float32).to(self._device)
            net_input = torch.cat([x_tensor * m_tensor, m_tensor], dim=1)
            x_imputed = self._model(net_input).cpu().numpy()

        result = X_arr.copy()
        nan_mask = np.isnan(result)
        result[nan_mask] = x_imputed[nan_mask]

        for j in range(result.shape[1]):
            still_nan = np.isnan(result[:, j])
            if still_nan.any():
                result[still_nan, j] = self._init_values[j]

        return pd.DataFrame(result, columns=self._columns, index=X_missing.index)
