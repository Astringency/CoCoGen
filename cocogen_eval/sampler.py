"""VP reverse-SDE imputation with RePaint (CoCoGen Appendix D).

Known values are sampled from the VP forward marginal. There are `steps`
positive-time score evaluations, each followed by `repaint` EXTRA forward /
reverse visits. Physical corrections use only the solution channel by default.
No unobserved ground truth is accepted by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time

import torch

from .common import sha_tensor


@dataclass(frozen=True)
class SamplerConfig:
    steps: int = 2000
    repaint: int = 1
    physics_steps: int = 50
    physics_post: int = 10
    max_physical_update: float = 2e-4
    correction_channels: str = 'solution'
    backtracks: int = 4
    forward_method: str = 'euler'
    final_denoise: bool = True
    seed: int = 0

    def __post_init__(self):
        if self.steps < 2 or self.repaint < 0 or min(self.physics_steps, self.physics_post) < 0:
            raise ValueError('Invalid time/repaint/physics step counts')
        if self.forward_method not in {'euler', 'exact'}:
            raise ValueError(self.forward_method)
        if self.correction_channels not in {'solution', 'all'}:
            raise ValueError(self.correction_channels)
        if self.max_physical_update < 0 or self.backtracks < 0:
            raise ValueError('Invalid physical step/backtrack size')


class PerSampleNoise:
    """Independent ID-seeded streams; batching does not change noise draws."""
    def __init__(self, ids, shape, device, *, seed=0, namespace='', tile=8):
        self.shape = tuple(shape)
        self.device = device
        self.tile = tile
        self.generators = []
        for sample_id in ids:
            digest = hashlib.sha256(f'{seed}:{namespace}:{sample_id}'.encode()).digest()
            value = int.from_bytes(digest[:8], 'little') % (2**63-1)
            self.generators.append(torch.Generator(device=device).manual_seed(value))
        self.position = tile
        self.buffer = None

    def draw(self):
        if self.position == self.tile:
            self.buffer = torch.stack([torch.randn((self.tile, *self.shape), device=self.device,
                                        generator=g) for g in self.generators])
            self.position = 0
        out = self.buffer[:, self.position].contiguous()
        self.position += 1
        return out


class Normalizer:
    def __init__(self, record, device):
        self.mean = torch.tensor(record['mean'], device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(record['std'], device=device).view(1, -1, 1, 1) + record['eps']

    def encode(self, x):
        return (x-self.mean)/self.std

    def decode(self, x):
        return x*self.std+self.mean


def physical_correction(x, mask, norm, residual_fn, config):
    """Autodiff of global residual norm, with per-example guarded descent.

The descent is in physical units, followed by the correct per-channel encode.
The max-absolute-gradient normalization and backtracking are numerical safeguards
around the paper's residual descent. `all` channels is an explicit extension.
"""
    with torch.enable_grad():
        physical = norm.decode(x.detach()).requires_grad_(True)
        residual = residual_fn(physical)
        loss = residual.flatten(1).square().mean(1)
        grad, = torch.autograd.grad(loss.sum(), physical)
    grad = grad.detach().masked_fill(mask, 0)
    if config.correction_channels == 'solution' and x.shape[1] > 1:
        grad[:, :-1] = 0
    finite = torch.isfinite(grad).flatten(1).all(1) & torch.isfinite(loss)
    grad = torch.nan_to_num(grad, nan=0., posinf=0., neginf=0.)
    maximum = grad.abs().flatten(1).amax(1).view(-1, 1, 1, 1)
    delta = config.max_physical_update * grad / maximum.clamp_min(1e-30)
    origin = physical.detach()
    accepted = ~finite
    best = origin.clone()
    evaluations = 1
    for _ in range(config.backtracks + 1):
        trial = origin-delta
        new_loss = residual_fn(trial).flatten(1).square().mean(1)
        evaluations += 1
        take = (~accepted) & torch.isfinite(new_loss) & (new_loss <= loss.detach())
        best = torch.where(take[:, None, None, None], trial, best)
        accepted |= take
        delta *= 0.5
    # Copy original normalized entries exactly where no update is allowed.
    corrected = norm.encode(best)
    return torch.where(grad != 0, corrected, x), evaluations


@torch.no_grad()
def sample(network, known_values, mask, ids, normalizer, sde, label, config,
           *, residual_fn=None, namespace='', progress=None):
    """Return physical predictions and a receipt. Only mask-covered values enter."""
    if known_values.shape != mask.shape or len(ids) != len(known_values):
        raise ValueError('Input shapes/IDs differ')
    mask = mask.bool()
    # Enforce the observation information boundary, even if a caller supplied a
    # dense array: every unobserved entry is zeroed before any normalization.
    known_values = torch.where(mask, known_values, 0.)
    if not torch.isfinite(known_values).all():
        raise ValueError('Nonfinite observed data')
    norm = Normalizer(normalizer, known_values.device)
    observed = norm.encode(known_values)
    rng = PerSampleNoise(ids, known_values.shape[1:], known_values.device,
                         seed=config.seed, namespace=namespace)
    beta_min, beta_max = float(sde['beta_min']), float(sde['beta_max'])
    dbeta = beta_max-beta_min
    def integral(t):
        return beta_min*t+0.5*dbeta*t*t
    def marginal(t):
        a = math.exp(-0.5*integral(t))
        sigma = math.sqrt(max(0., -math.expm1(-integral(t))))
        return a*observed+sigma*rng.draw()
    if known_values.is_cuda:
        torch.cuda.synchronize(known_values.device)
        torch.cuda.reset_peak_memory_stats(known_values.device)
    start = time.monotonic()
    x = rng.draw()
    initial_sha = sha_tensor(x)
    x = torch.where(mask, marginal(1.), x)
    c = torch.full((len(ids), network.cond_size), float(label), device=x.device)
    context_mask = torch.zeros_like(c)
    nfe = 0
    residual_evaluations = 0
    correction_count = 0
    h = 1./config.steps
    for i in range(config.steps):
        t = (config.steps-i)/config.steps
        next_t = (config.steps-i-1)/config.steps
        time_batch = torch.full((len(ids),), t, device=x.device)
        beta = beta_min+dbeta*t
        for visit in range(config.repaint+1):
            score = network(x, c, time_batch, context_mask)
            nfe += 1
            mean = x+h*beta*(0.5*x+score)
            noise = rng.draw()
            # The original CoCoGen returns the last predictor mean. Keep this
            # denoising convention; the same draw is consumed in either mode.
            if config.final_denoise and i == config.steps-1 and visit == config.repaint:
                x = mean
            else:
                x = mean+math.sqrt(beta*h)*noise
            x = torch.where(mask, marginal(next_t), x)
            if residual_fn is not None and config.max_physical_update and i >= max(0, config.steps-config.physics_steps):
                x, evaluations = physical_correction(x, mask, norm, residual_fn, config)
                residual_evaluations += evaluations
                correction_count += 1
            if visit < config.repaint:
                if config.forward_method == 'euler':
                    beta_forward = beta_min+dbeta*next_t
                    x = x-0.5*h*beta_forward*x+math.sqrt(beta_forward*h)*rng.draw()
                else:
                    interval = integral(t)-integral(next_t)
                    x = math.exp(-0.5*interval)*x+math.sqrt(-math.expm1(-interval))*rng.draw()
                x = torch.where(mask, marginal(t), x)
        if progress and ((i+1) % 250 == 0 or i == config.steps-1):
            progress(dict(step=i+1, steps=config.steps, nfe=nfe))
    x = torch.where(mask, observed, x)
    if residual_fn is not None and config.max_physical_update:
        for _ in range(config.physics_post):
            x, evaluations = physical_correction(x, mask, norm, residual_fn, config)
            residual_evaluations += evaluations
            correction_count += 1
    physical = torch.where(mask, known_values, norm.decode(x))
    if not torch.isfinite(physical).all():
        raise FloatingPointError('Nonfinite final sample; no finite-score substitution is allowed')
    if known_values.is_cuda:
        torch.cuda.synchronize(known_values.device)
    receipt = dict(config=asdict(config), nfe=nfe, correction_count=correction_count,
        residual_evaluations=residual_evaluations, initial_noise_sha256=initial_sha,
        noise_stream='sha256(seed:namespace:sample_id), per-ID torch generator, tile=8',
        namespace=namespace, seconds=time.monotonic()-start,
        peak_bytes=torch.cuda.max_memory_allocated(known_values.device) if known_values.is_cuda else 0,
        min_score_time=h, final_time=0., label=label,
        algorithm='CoCoGen Appendix D reverse VP SDE / Euler-Maruyama / RePaint')
    return physical, receipt
