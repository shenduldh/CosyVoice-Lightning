import torch
from torch import nn, Tensor

@torch.jit.script
def snake_fast_inference(x: Tensor, a: Tensor, inv_2b: Tensor) -> Tensor:
    return x + (1.0 - torch.cos(2.0 * a * x)) * inv_2b

class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        init_val = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        self.alpha = nn.Parameter(init_val * alpha)
        self.alpha.requires_grad = alpha_trainable
        
        # persistent=False makes these invisible to the state_dict
        self.register_buffer('a_eff', torch.ones(1, in_features, 1), persistent=False)
        self.register_buffer('inv_2a', torch.ones(1, in_features, 1), persistent=False)
        self._is_prepared = False

    def prepare(self):
        with torch.no_grad():
            a = torch.exp(self.alpha) if self.alpha_logscale else self.alpha
            a = a.view(1, -1, 1)
            self.a_eff.copy_(a)
            self.inv_2a.copy_(1.0 / (2.0 * a + 1e-9))
        self._is_prepared = True

    def forward(self, x: Tensor) -> Tensor:
        if not self._is_prepared and not self.training:
            self.prepare()
        
        if not self.training:
            return snake_fast_inference(x, self.a_eff, self.inv_2a)
        
        a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
        return x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * a + 1e-9)

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        init_val = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        
        self.alpha = nn.Parameter(init_val * alpha)
        self.beta = nn.Parameter(init_val * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        # persistent=False fixes the Missing Key error
        self.register_buffer('a_eff', torch.ones(1, in_features, 1), persistent=False)
        self.register_buffer('inv_2b', torch.ones(1, in_features, 1), persistent=False)
        self._is_prepared = False

    def prepare(self):
        with torch.no_grad():
            a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
            b = (torch.exp(self.beta) if self.alpha_logscale else self.beta).view(1, -1, 1)
            self.a_eff.copy_(a)
            self.inv_2b.copy_(1.0 / (2.0 * b + 1e-9))
        self._is_prepared = True

    def forward(self, x: Tensor) -> Tensor:
        if not self._is_prepared and not self.training:
            self.prepare()

        if not self.training:
            return snake_fast_inference(x, self.a_eff, self.inv_2b)
        
        a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
        b = (torch.exp(self.beta) if self.alpha_logscale else self.beta).view(1, -1, 1)
        return x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * b + 1e-9)
