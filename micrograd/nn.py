'''
Módulos neurais reutilizáveis construídos sobre e engine de tensores.
'''
import numpy as np
from typing import List
from .engine import Tensor

class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None
            
    def parameters(self) -> List[Tensor]:
        return []
    
class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        # Inicialização He (Kaiming Normal) para mitigar vanishing/exploding gradients
        std = np.sqrt(2. / in_features)
        self.weight = Tensor(np.random.randn(in_features, out_features) * std, requires_grad = True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad = True)
        
    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.weight) + self.bias
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]
    
class Sequential(Module):
    def __init__(self, * layers: Module):
        self.layers = list(layers)
        
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
