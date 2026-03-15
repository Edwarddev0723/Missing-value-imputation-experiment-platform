"""MIWAE — Missing data Importance-Weighted Autoencoder (D2).

Mattei & Friston, 2019.
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


@METHOD_REGISTRY.register("miwae")
class MIWAEImputer(BaseImputer):
    """MIWAE: Importance-Weighted Autoencoder for missing data — D2."""

    name = "miwae"
    family = FAMILY_DL

    def __init__(
        self,
        latent_dim: int = 32,
        n_hidden: int = 128,
        n_samples: int = 20,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        device: str = "cpu",
        early_stopping_patience: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            n_hidden=n_hidden,
            n_samples=n_samples,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device,
            early_stopping_patience=early_stopping_patience,
            **kwargs,
        )
        self.latent_dim = latent_dim
        self.n_hidden = n_hidden
        self.n_samples = n_samples
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device_request = device
        self.early_stopping_patience = early_stopping_patience
        self._encoder = None
        self._decoder_mu = None
        self._decoder_logvar = None
        self._columns: list[str] = []
        self._init_values: np.ndarray | None = None

    def _build_model(self, input_dim: int):
        import torch.nn as nn

        # Encoder: x -> (z_mu, z_logvar)
        encoder = nn.Sequential(
            nn.Linear(input_dim, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
        )
        enc_mu = nn.Linear(self.n_hidden, self.latent_dim)
        enc_logvar = nn.Linear(self.n_hidden, self.latent_dim)

        # Decoder: z -> (x_mu, x_logvar)
        decoder_body = nn.Sequential(
            nn.Linear(self.latent_dim, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
        )
        dec_mu = nn.Linear(self.n_hidden, input_dim)
        dec_logvar = nn.Linear(self.n_hidden, input_dim)

        return encoder, enc_mu, enc_logvar, decoder_body, dec_mu, dec_logvar

    def fit(self, X_train: pd.DataFrame, metadata: dict[str, Any] | None = None) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._columns = list(X_train.columns)
        device = get_device(self.device_request)
        torch_device = torch.device(device)

        X_arr = X_train.values.astype(np.float32)
        obs_mask = ~np.isnan(X_arr)
        self._init_values = np.nanmean(X_arr, axis=0)
        fill_row = self._init_values[np.newaxis, :]
        X_filled = np.where(~obs_mask, fill_row, X_arr)

        input_dim = X_filled.shape[1]
        encoder, enc_mu, enc_logvar, decoder_body, dec_mu, dec_logvar = self._build_model(input_dim)

        all_modules = nn.ModuleList([encoder, enc_mu, enc_logvar, decoder_body, dec_mu, dec_logvar])
        all_modules = all_modules.to(torch_device)

        params = list(all_modules.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        X_tensor = torch.tensor(X_filled, dtype=torch.float32)
        M_tensor = torch.tensor(obs_mask, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, M_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        patience_counter = 0

        _amp_dtype = torch.float16
        _use_amp = 'mps' not in str(torch_device)
        _amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_m in loader:
                batch_x = batch_x.to(torch_device)
                batch_m = batch_m.to(torch_device)
                batch_size = batch_x.shape[0]

                with torch.autocast(device_type=_amp_device, dtype=_amp_dtype, enabled=_use_amp):
                    # Encode
                    h = encoder(batch_x)
                    z_mu = enc_mu(h)
                    z_logvar = enc_logvar(h)

                    # Sample K latent codes
                    K = self.n_samples
                    z_mu_k = z_mu.unsqueeze(1).expand(-1, K, -1)
                    z_logvar_k = z_logvar.unsqueeze(1).expand(-1, K, -1)
                    eps = torch.randn_like(z_mu_k)
                    z = z_mu_k + torch.exp(0.5 * z_logvar_k) * eps  # (B, K, D_z)

                    # Decode
                    z_flat = z.reshape(-1, self.latent_dim)
                    h_dec = decoder_body(z_flat)
                    x_mu = dec_mu(h_dec).reshape(batch_size, K, input_dim)
                    x_logvar = dec_logvar(h_dec).reshape(batch_size, K, input_dim)

                    # Log-likelihood (only on observed)
                    batch_x_k = batch_x.unsqueeze(1).expand(-1, K, -1)
                    batch_m_k = batch_m.unsqueeze(1).expand(-1, K, -1)
                    log_p_x_given_z = -0.5 * (
                        x_logvar + (batch_x_k - x_mu) ** 2 / (torch.exp(x_logvar) + 1e-8)
                    )
                    log_p_x_given_z = (log_p_x_given_z * batch_m_k).sum(-1)

                    # KL divergence
                    log_p_z = -0.5 * (z ** 2).sum(-1)
                    log_q_z = -0.5 * (z_logvar_k + (z - z_mu_k) ** 2 / (torch.exp(z_logvar_k) + 1e-8)).sum(-1)

                    # IWAE bound
                    log_w = log_p_x_given_z + log_p_z - log_q_z
                    log_w_max = log_w.max(dim=1, keepdim=True)[0]
                    w = torch.exp(log_w - log_w_max)
                    loss = -(log_w_max.squeeze() + torch.log(w.mean(dim=1) + 1e-10)).mean()

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
                    logger.debug(f"MIWAE early stopping at epoch {epoch + 1}")
                    break

        self._encoder = encoder
        self._enc_mu = enc_mu
        self._enc_logvar = enc_logvar
        self._decoder_body = decoder_body
        self._dec_mu = dec_mu
        self._device = torch_device
        self._input_dim = input_dim

    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        import torch

        if self._encoder is None:
            raise RuntimeError("MIWAE must be fit before impute.")

        X_arr = X_missing.values.astype(np.float32)
        nan_mask = np.isnan(X_arr)
        fill_row = self._init_values[np.newaxis, :]
        X_filled = np.where(nan_mask, fill_row, X_arr)

        self._encoder.eval()
        self._enc_mu.eval()
        self._decoder_body.eval()
        self._dec_mu.eval()
        _use_amp = 'mps' not in str(self._device)
        _amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with torch.inference_mode(), torch.autocast(device_type=_amp_device, dtype=torch.float16, enabled=_use_amp):
            X_tensor = torch.tensor(X_filled, dtype=torch.float32, device=self._device)
            h = self._encoder(X_tensor)
            z_mu = self._enc_mu(h)
            h_dec = self._decoder_body(z_mu)
            x_mu = self._dec_mu(h_dec).cpu().numpy()

        result = X_arr.copy()
        result[nan_mask] = x_mu[nan_mask]
        still_nan = np.isnan(result)
        if still_nan.any():
            fill_row = self._init_values[np.newaxis, :]
            result = np.where(still_nan, fill_row, result)

        return pd.DataFrame(result, columns=self._columns, index=X_missing.index)
