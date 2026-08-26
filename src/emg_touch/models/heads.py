from __future__ import annotations

import math

import torch
from torch import nn


class MDNHead(nn.Module):
    def __init__(self, input_dim: int, components: int) -> None:
        super().__init__()
        self.components = components
        self.projection = nn.Linear(input_dim, components * 6)

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.projection(context).reshape(-1, self.components, 6)
        return {
            "logits": raw[..., 0],
            "means": torch.sigmoid(raw[..., 1:3]),
            "log_scales": raw[..., 3:5].clamp(-6.0, 2.0),
            "correlations": torch.tanh(raw[..., 5]).clamp(-0.95, 0.95),
        }

    @staticmethod
    def expected_value(distribution: dict[str, torch.Tensor]) -> torch.Tensor:
        weights = torch.softmax(distribution["logits"], dim=-1)
        return torch.sum(weights.unsqueeze(-1) * distribution["means"], dim=1)

    @staticmethod
    def sample(distribution: dict[str, torch.Tensor], count: int) -> torch.Tensor:
        logits = distribution["logits"]
        component_indices = torch.distributions.Categorical(logits=logits).sample((count,)).transpose(0, 1)
        batch = logits.size(0)
        means = distribution["means"]
        scales = distribution["log_scales"].exp()
        rho = distribution["correlations"]
        gather2 = component_indices.unsqueeze(-1).expand(batch, count, 2)
        chosen_mean = means.gather(1, gather2)
        chosen_scale = scales.gather(1, gather2)
        chosen_rho = rho.gather(1, component_indices)
        noise = torch.randn(batch, count, 2, device=logits.device)
        x = noise[..., 0]
        y = chosen_rho * x + torch.sqrt(1.0 - chosen_rho.square()) * noise[..., 1]
        correlated = torch.stack([x, y], dim=-1)
        return (chosen_mean + chosen_scale * correlated).clamp(0.0, 1.0)


class ConditionalCVAEHead(nn.Module):
    def __init__(self, context_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.prior = nn.Sequential(
            nn.Linear(context_dim, context_dim), nn.GELU(), nn.Linear(context_dim, latent_dim * 2)
        )
        self.posterior = nn.Sequential(
            nn.Linear(context_dim + 2, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(context_dim + latent_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim // 2),
            nn.GELU(),
            nn.Linear(context_dim // 2, 4),
        )

    @staticmethod
    def _split(parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_variance = parameters.chunk(2, dim=-1)
        return mean, log_variance.clamp(-8.0, 4.0)

    @staticmethod
    def _reparameterize(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)

    def _decode(self, context: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.decoder(torch.cat([context, latent], dim=-1))
        mean = torch.sigmoid(raw[..., :2])
        log_scale = raw[..., 2:].clamp(-6.0, 2.0)
        return mean, log_scale

    def forward(
        self, context: torch.Tensor, target: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        prior_mean, prior_logvar = self._split(self.prior(context))
        if self.training and target is not None:
            posterior_mean, posterior_logvar = self._split(
                self.posterior(torch.cat([context, target], dim=-1))
            )
            latent = self._reparameterize(posterior_mean, posterior_logvar)
        else:
            posterior_mean, posterior_logvar = prior_mean, prior_logvar
            latent = prior_mean
        mean, log_scale = self._decode(context, latent)
        return {
            "mean": mean,
            "log_scale": log_scale,
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "posterior_mean": posterior_mean,
            "posterior_logvar": posterior_logvar,
        }

    def sample(self, context: torch.Tensor, count: int) -> torch.Tensor:
        prior_mean, prior_logvar = self._split(self.prior(context))
        batch = context.size(0)
        noise = torch.randn(batch, count, self.latent_dim, device=context.device)
        latent = prior_mean[:, None] + torch.exp(0.5 * prior_logvar[:, None]) * noise
        expanded_context = context[:, None].expand(-1, count, -1)
        mean, log_scale = self._decode(
            expanded_context.reshape(batch * count, -1), latent.reshape(batch * count, -1)
        )
        observation = mean + log_scale.exp() * torch.randn_like(mean)
        return observation.reshape(batch, count, 2).clamp(0.0, 1.0)


def mdn_negative_log_likelihood(
    distribution: dict[str, torch.Tensor], target: torch.Tensor
) -> torch.Tensor:
    means = distribution["means"]
    scales = distribution["log_scales"].exp()
    rho = distribution["correlations"]
    standardized = (target[:, None] - means) / scales
    x, y = standardized.unbind(dim=-1)
    one_minus_rho2 = (1.0 - rho.square()).clamp_min(1e-5)
    quadratic = (x.square() - 2.0 * rho * x * y + y.square()) / one_minus_rho2
    log_normalizer = (
        math.log(2.0 * math.pi)
        + distribution["log_scales"].sum(dim=-1)
        + 0.5 * torch.log(one_minus_rho2)
    )
    component_log_prob = -0.5 * quadratic - log_normalizer
    mixture_log_prob = torch.logsumexp(
        torch.log_softmax(distribution["logits"], dim=-1) + component_log_prob,
        dim=-1,
    )
    return -mixture_log_prob.mean()


def cvae_loss(
    distribution: dict[str, torch.Tensor], target: torch.Tensor, beta: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scale = distribution["log_scale"].exp()
    reconstruction = (
        0.5 * ((target - distribution["mean"]) / scale).square()
        + distribution["log_scale"]
        + 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=-1).mean()
    qm, qlv = distribution["posterior_mean"], distribution["posterior_logvar"]
    pm, plv = distribution["prior_mean"], distribution["prior_logvar"]
    kl = 0.5 * (
        plv - qlv + (qlv.exp() + (qm - pm).square()) / plv.exp() - 1.0
    ).sum(dim=-1).mean()
    return reconstruction + beta * kl, {"reconstruction": reconstruction, "kl": kl}

