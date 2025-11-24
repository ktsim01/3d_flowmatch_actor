import torch
import numpy as np


class RFScheduler:
    """
    Code based on:
    https://github.com/cloneofsimo/minRF/blob/main/advanced/main_t2i.py
    """

    def __init__(self, noise_sampler="logit_normal", noise_sampler_config={}):
        self.noise_sampler = noise_sampler
        self.noise_sampler_config = noise_sampler_config

    def set_timesteps(self, num_inference_steps, device='cpu'):
        self.timesteps = torch.from_numpy(
            np.arange(num_inference_steps + 1)[1:][::-1].astype(np.float32)
            / num_inference_steps
        ).to(device)
        # Precompute previous timesteps, where prev_t[0] = 0
        self.prev_timesteps = torch.cat((
            self.timesteps[1:],
            torch.zeros(1, device=device, dtype=self.timesteps.dtype)
        ))

    def sample_noise_step(self, num_noise, device):
        if self.noise_sampler == "uniform":
            samples = torch.full((num_noise,), 0, dtype=torch.float32,
                                 device=device)
            timesteps = samples.uniform_()
        elif self.noise_sampler == "logit_normal":
            samples = torch.full((num_noise,), 0, dtype=torch.float32,
                                 device=device)
            samples = samples.normal_(
                mean=self.noise_sampler_config['mean'],
                std=self.noise_sampler_config['std']
            )
            timesteps = torch.sigmoid(samples)
        elif self.noise_sampler == "pi0":
            alpha, beta = 1.5, 1.0
            timesteps = torch.distributions.Beta(
                alpha, beta
            ).sample((num_noise,)).to(device).clamp(max=0.999)
        else:
            raise NotImplementedError(f"{self.noise_sampler} not implemented")

        return timesteps

    def add_noise(self, original_samples, noise, timesteps):
        x = original_samples
        z1 = noise
        t = timesteps
        b = x.size(0)
        
        # Interpolate between Z0 and Z1
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        zt = (1 - texp) * x + texp * z1
        return zt.to(x.dtype)

    def step(self, model_output, timestep_ind, sample):
        zt = sample
        vc = model_output

        timestep = self.timesteps[timestep_ind].to(vc.device)
        prev_t = self.prev_timesteps[timestep_ind].to(vc.device)
        dt = timestep - prev_t
        pred_prev_sample = zt - dt * vc  # z_t'

        return DummyClass(prev_sample=pred_prev_sample)

    def prepare_target(self, noise, gt):
        return noise - gt

class LocalRFScheduler:
    def __init__(self, sigma=0.01):   # std of local diffusion
        self.sigma = sigma

    def set_timesteps(self, num_inference_steps, device='cpu'):
        self.timesteps = torch.linspace(1, 0, num_inference_steps, device=device)
        self.prev_timesteps = torch.cat((self.timesteps[1:], torch.zeros(1, device=device)))

    def sample_noise_step(self, num_noise, device):
        # Uniform t in [0,1]
        return torch.rand(num_noise, device=device)

    def add_noise(self, x0, noise, timesteps):
        b = x0.size(0)
        t = timesteps.view(b, *([1] * (x0.dim() - 1)))
        return x0 + t * self.sigma * noise

    def prepare_target(self, noise, gt):
        # v = sigma * eps
        return self.sigma * noise

    def step(self, model_output, timestep_ind, sample):
        # reverse ODE integration
        t = self.timesteps[timestep_ind]
        prev_t = self.prev_timesteps[timestep_ind]
        dt = t - prev_t
        prev_sample = sample - dt * (self.sigma * model_output)
        return DummyClass(prev_sample=prev_sample)

class DummyClass:

    def __init__(self, prev_sample):
        self.prev_sample = prev_sample
