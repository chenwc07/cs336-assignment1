import torch
from torch import nn
import math
from jaxtyping import Bool, Float, Int
from torch import Tensor

from collections.abc import Callable, Iterable
from typing import Optional


def cross_entropy_loss(
        inputs: Float[Tensor, " batch_size vocab_size"],
        targets: Int[Tensor, " batch_size"]
    ) -> Float[Tensor, ""]:
    
    # log_probs = torch.log_softmax(inputs, dim=-1) # 官方实现
    # log_probs = torch.log(torch.softmax(inputs, dim=-1)) # 不符合数值稳定性，即使inputs减去最大值，在log时仍会出现下溢（log(exp(-1000)) = log(0) = -inf）
    # log_probs = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True) # 数值稳定性实现

    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    log_probs = inputs - torch.log(torch.sum(torch.exp(inputs), dim=-1, keepdim=True)) # 数值稳定性实现+不使用封装函数

    batch_size = inputs.shape[0]
    # loss = -log_probs[torch.arange(batch_size), targets].mean()
    loss = -torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).mean()
    return loss

class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state['t'] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params: Iterable[torch.nn.Parameter], 
        lr: float = 1e-3, 
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        # 防御性编程
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))
                t = state.get('t', 0)
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                lr_t = lr * math.sqrt(1 - beta2 ** (t + 1)) / (1 - beta1 ** (t + 1))
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state['m'] = m
                state['v'] = v
                state['t'] = t + 1
        return loss


def cosine_learning_rate_schedule(
        t: int,
        lr_max: float,
        lr_min: float,
        T_warmup: int,
        T_cos: int
) -> float:
    if t < T_warmup:
        lr = lr_max * t / T_warmup
    elif T_warmup <= t <= T_cos:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - T_warmup) / (T_cos - T_warmup)))
    else:
        lr = lr_min
    return lr

def gradient_clipping(
        parameters: Iterable[torch.nn.Parameter],
        max_norm: float,
):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    return total_norm
    
if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.