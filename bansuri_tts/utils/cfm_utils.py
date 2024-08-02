import ot
import torch
from bansuri_tts.utils.optimal_transport import OTPlanSampler

class ConditionalFlowMatcher:
    def __init__(self, sigma=0.0):
        self.sigma = sigma
    
    def sample_noise_like(self, x1):
        return torch.randn_like(x1)

    def sample_t(self, x1):
        t = torch.rand((x1.size(0),)).to(x1)
        t = t.reshape(-1, *([1] * (x1.dim() - 1)))
    
    def sample_location(self, x0, x1, t):
        return (1 - (1 - self.sigma) * t) * x0 + x1 * t
    
    def compute_conditional_flow(self, x0, x1):
        return x1 - (1 - self.sigma) * x0
    
    def sample_location_and_conditional_flow(self, x1, x0=None, t=None):
        if x0 is None:
            x0 = self.sample_noise_like(x1)
        if t is None:
            t = self.sample_t(x1)
        xt = self.sample_location(x0, x1, t)
        ut = self.compute_conditional_flow(x0, x1)

        return t, x0, xt, ut

class OTConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self, sigma=0.0, method="exact"):
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(method=method)

    def sample_location_and_conditional_flow(self, x1, x0=None, t=None):
        if x0 is None:
            x0 = self.sample_noise_like(x1)
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x1, x0, t)
    
    def guided_sample_location_and_conditional_flow(self, x1, x0=None, y1=None, y0=None, t=None):
        if x0 is None:
            x0 = self.sample_noise_like(x1)
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)

        t, x0, xt, ut = super().sample_location_and_conditional_flow(x1, x0, t)

        return t, x0, xt, ut, y0, y1