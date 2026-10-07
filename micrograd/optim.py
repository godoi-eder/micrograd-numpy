'''
Otimizadores estocásticos com atualização manual dos buffers de tensores.
'''
import numpy as np
from typing import List
from .engine import Tensor

class Optimizer:
    def __init__(self, params: List[Tensor]):
        self.params = [p for p in params if p.requires_grad]
        
    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None
            
    def step(self) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.params]
        
    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.velocities[i] = self.momentum * self.velocities[i] + self.lr * p.grad
            p.data -= self.velocities[i]
            
class Adam(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0
    
    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            # Estimativa de primeiro momento (média)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            
            # Estimativa de segundo momento (variância não centrada)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            
            # Correção de viés inicial
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
