'''
Função de custo baseados exclusivamente em operações NumPy/Tensor.
'''
import numpy as np
from .engine import Tensor

def mse_loss(preds: Tensor, targets: Tensor) -> Tensor:
    '''Mean Squared Error: L = 1/N * sum((y_hat - y)^2)'''
    diff = preds - targets
    squared = diff * diff
    return squared.sum() * (1.0 / preds.data.size)

def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    '''
    Cross Entropy Loss com estabilização pelo truque do Log-Sum-Exp:
    log_sum_exp(z) = C + log(sum(exp(z - c)))
    '''
    # targets: one-hot encoded matrix
    max_val = np.max(logits.data, axis = 1, keepdims = True)
    stabilized_logits = logits - Tensor(max_val)
    exp_logits = stabilized_logits.exp()
    sum_exp = exp_logits.sum(axis = 1, keepdims =  True)
    log_probs = stabilized_logits - sum_exp.log()
    
    # Perda média NLL: - (targets * logs_probs)
    nll = -(targets * log_probs).sum() * (1.0 / logits.data.shape[0])
    return nll
