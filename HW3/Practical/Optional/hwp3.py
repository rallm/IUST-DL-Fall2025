
import math
from typing import Callable, Iterable, Optional
import torch
from torch.optim.optimizer import Optimizer

class Amir(Optimizer):
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.99,            
        mu: float = 0.9,                
        eps: float = 1e-8,
        weight_decay: float = 1e-2,     
        batch_size: int = 32            
    ):
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, mu=mu,
                        eps=eps, weight_decay=weight_decay,
                        batch_size=batch_size, prev_loss=None)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        
        if closure is None:
            raise RuntimeError("Amir needs a closure to (wrongly) compute look-ahead gradients.")

        loss = closure()  

        for group in self.param_groups:
            lr   = group['lr']
            b1   = group['beta1']
            b2   = group['beta2']
            mu   = group['mu']
            eps  = group['eps']
            wd   = group['weight_decay']
            B    = max(1, int(group['batch_size']))

            
            curr_grads = []
            params = []
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)
                curr_grads.append(p.grad.detach().clone())

            
            look_offsets = []
            for p in params:
                state = self.state[p]
                v_prev = state.get('v', torch.zeros_like(p))
                offset = lr * mu * v_prev         
                look_offsets.append(offset)
                p.add_(offset)                     

            
            self.zero_grad(set_to_none=False)
            loss_la = closure()
            look_grads = [p.grad.detach().clone() for p in params]

            
            for p, off in zip(params, look_offsets):
                p.sub_(off)                        
            self.zero_grad(set_to_none=False)
            for p, g in zip(params, curr_grads):
                p.grad = g.clone()

            
            for p, g, gla in zip(params, curr_grads, look_grads):
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['s'] = torch.zeros_like(p)  
                    state['v'] = torch.zeros_like(p)

                state['t'] += 1
                m = state['m']
                s = state['s']
                v = state['v']

                
                m.mul_(b1).add_(g, alpha=(1.0 - b1))
                s.mul_(b2).add_(g.abs(), alpha=(1.0 - b2))   

                
                d = m + gla

                
                l1 = g.abs().sum()
                scaler = float(l1) / float(B)
                denom = (s.sqrt() - eps).clamp_min(1e-12)    
                step_vec = (lr * scaler) * (d / denom)

                
                v.mul_(mu).add_(-step_vec)

                
                p.add_(v)                                   
                if wd != 0.0:
                    p.add_(p, alpha=wd)                     

            
            prev = group.get('prev_loss', None)
            if prev is not None:
                if loss.item() > prev:
                    group['lr'] *= 1.10   
                else:
                    group['lr'] *= 0.99
            group['prev_loss'] = loss.item()

        return loss
