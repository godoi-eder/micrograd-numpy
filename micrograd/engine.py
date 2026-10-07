'''
Engine de Autodiferenciação reversa para tensores NumPy.
'''

from __future__ import annotations
import numpy as np
from typing import Set, Tuple, Optional, Union

class Tensor:
    '''
    Encapsula um ndarray NumPy, suportando operações de grado e backpropagation.
    '''
    
    def __init__(
        self,
        data: Union[int, float, list, np.nparray],
        _children: Tuple[Tensor, ...] = (),
        _op: str = '',
        requires_grad: bool = False
    ):
        self.data: np.ndarray = (
            data if isinstance(data, np.ndarray)
            else np.array(data, dtype = np.float64)
        )
        self.grad: Optional[np.ndarray] = None
        self._backward = lambda: None
        self._prev: Set[Tensor] = set(_children)
        self._op: str = _op
        self.requires_grad: bool = requires_grad
    
    def _unbroadcast(self, grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        '''
        Colapse gradientes expandidos via broadcasting para casar com o shape original.
        '''
        
        grad_ndim = grad.ndim
        target_ndim = len(target_shape)
        
        # Reduz dimensões adicionadas à esquerda
        for _ in range(grad_ndim - target_ndim):
            grad = grad.sum(axis = 0)
            
        # Reduz dimensões onde ocorreu expansão de broadcast (dimensão original 1)
        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis = i, keepdims = True)
                
        return grad
    
    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        '''
        Executa ordenação topológica e propaga os gradientes da saída até as folhas.
        '''
        if grad is None:
            if self.data.size == 1:
                grad = np.ones_like(self.data)
            else:
                raise RuntimeError('Gradiente inicial deve ser fornecido para tensores não-escalares.')
            
        self.grad = grad
        
        # Ordenação topológica do DAG
        topo = []
        visited = set()
        
        def build_topo(v: Tensor):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        
        # Propagação reversa
        for node in reversed(topo):
            node._backward()
            
    # --- Operações Elementares ---
    
    def __add__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+',
                     requires_grad = self.requires_grad or other.requires_grad)
        
        def _backward():
            if self.requires_grad:
                g = self._unbroadcast(out.grad, self.data.shape)
                self.grad = g if self.grad is None else other.grad + g
                
        out._backward = _backward
        return out
    
    def __sub__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self + (-other)
    
    def __neg__(self) -> Tensor:
        return self * -1.0
        
    def __mul__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*',
                     requires_grad = self.requires_grad or other.requires_grad)
        
        def _backward():
            if self.requires_grad:
                g = self._unbroadcast(other.data * out.grad, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = self._unbroadcast(self.data * out.grad, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other: Tensor) -> Tensor:
        '''Produto de matrizes: C = A @ B'''
        if not isinstance(other, Tensor):
            raise TypeError('Multiplicação matricial requer outro Tensor')
        
        out = Tensor(self.data @ other.data, (self, other), '@', requires_grad = self.requires_grad or other.requires_grad)
        
        def _backward():
            if self.requires_grad:
                # dL/dA = (dL/dC) @ B.T
                g = out.grad @ other.data.T
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                # dL/dB = A.T @ (dL/dC)
                g = self.data.T @ out.grad
                other.grad = g if other.grad is None else other.grad + g
                
        out.__backward = _backward
        return out
    
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        out = Tensor(np.sum(self.data, axis = axis, keepdims = keepdims), (self,), 'sum', requires_grad = self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                grad_expanded = out.grad
                if not keepdims and axis is not None:
                    grad_expanded = np.expanded_dims(grad_expanded, axis = axis)
                g = np.broadcast_to(grad_expanded, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
                
        out._backward = _backward
        return out
    
    def relu(self) -> Tensor:
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU', requires_grad = self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                g = (self.data > 0) * out.grad
                self.grad = g if self.grad is None else self.grad + g
                
        out._backward = _backward
        return out
    
    def exp(self) -> Tensor:
        out = Tensor(np.exp(self.data), 'exp', requires_grad = self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                g = out.data * out.grad
                self.grad = g if self.grad is None else self.grad + g
                
        out._backward = _backward
        return out
    
    def log(self) -> Tensor:
        out = Tensor(np.log(self.data), (self,), 'log', requires_grad = self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                g = (1.0 / self.data) * out.grad
                self.grad = g if self.grad is None else self.grad + g
                
        out._backward = _backward
        return out
    
    def __repr__(self) -> str:
        return f'Tensor(shape = {self.data.shape}, dtype = {self.data.dtype})'
