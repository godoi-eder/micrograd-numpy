'''
Execução de treinamento de ponta a ponta com MicroGrad-NumPy.
Classificação não-linear de dois anéis concêntricos (problema não separável linearmente).
'''
import numpy as np
from micrograd import Tensor, Linear, Sequential, cross_entropy_loss, Adam

def generate_synthetic_data(n_samples: int = 400):
    '''Gera dadso de dois círculos concêntricos usando coordenadas polares.'''
    np.random.seed(42)
    samples_per_ring = n_samples // 2
    
    # Anel interno (Classe 0)
    r1 = np.random.uniform(0.0, 0.4, samples_per_ring)
    theta1 = np.random.uniform(0.0, 2 * np.pi, samples_per_ring)
    x1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis = 1)
    y1 = np.zeros(samples_per_ring, dtype = int)
    
    # Anel externo (Classe 1)
    r2 = np.random.uniform(0.6, 1.0, samples_per_ring)
    theta2 = np.random.uniform(0.6, 1.0, samples_per_ring)
    x2 = np.stack([r2 * np.cos(theta2), r2 * np.sin(theta2)], axis = 1)
    y2 = np.ones(samples_per_ring, dtype = int)
    
    x = np.vstack([x1, x2])
    y = np.concatenate([y1, y2])
    
    # One-hot encoding
    y_one_hot = np.zeros((n_samples, 2))
    y_one_hot[np.arange(n_samples), y] = 1.0
    
    return x, y, y_one_hot

def run_pipeline():
    x_raw, y_raw, y_one_hot = generate_synthetic_data(n_samples = 600)
    
    # Conversão para tensores do MicroGrad
    x = Tensor(x_raw)
    y_target = Tensor(y_one_hot)
    
    # Definição da Arquitetura: 2 entradas -> 16 neurônios -> 8 neurônios -> 2 saídas
    layer1 = Linear(2, 16)
    layer2 = Linear(16, 8)
    layer3 = Linear(8, 2)
    
    params = layer1.parameters() + layer2.parameters() + layer3.parameters()
    optimizer = Adam(params, lr = 0.03)
    
    print('=' * 60)
    print('Iniciando Treinamento Neural (Autograd + Adam puro NumPy)...')
    print('=' * 60)
    
    for epoch in range(1, 101):
        # 1. Forwad Pass com ativações manuais
        h1 = layer1(x).relu()
        h2 = layer2(h1).relu()
        logits = layer3(h2)
        
        # 2. Cálculo da Função de Perda
        loss = cross_entropy_loss(logits, y_target)
        
        # 3. Backward Pass (Cálculo automático de gradientes no DAG)
        optimizer.zero_grad()
        loss.backward()
        
        # 4. Passo de Otimização dos Pesos (Adam update)
        optimizer.step()
        
        # 5. Métricas de Acurácia
        if epoch % 10 == 0 or epoch == 1:
            preds = np.argmax(logits.data, axis = 1)
            acc = np.mean(preds == y_raw) * 100.0
            print(f'Época {epoch:03d}/100 | Custo (Loss): {loss.data:.4f} | Acurácia: {acc:.2f}%')
            
    print('\nTreinamento concluído com convergência comprovada!')
    
if __name__ == '__main__':
    run_pipeline
