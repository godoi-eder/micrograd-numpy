# **MicroGrad-NumPy**

Uma engine modular e completa de **Difernciação Automática Reversa (Autograd)** e biblioteca de Deep Learnig implementada **do zero, utilizando exclusivamente a biblioteca NumPy**.

Desenvolvido para evidenciar conceitos avançados de computação tensorial, ordenação topológica de grafas direcionados acíclicos (DAF), backpropagation manual com broadcast
analítico e algorítmos de otimização estocástica de primeira ordem.

---

## **Destaques de Arquiteturas**

- **DAG Engine:** Ratreamento dinâmico de dependências e retropropagação de gradientes utilizando `Tensor`.
- **Broadcasting Analítico:** Método `_unbroadcast` que calcula e colapsa as dimensões expandidas por regras do NumPy durante o `bachward()`.
- **Estabilidade Numérica:** Formulação da perda por Entropia Cruzada usando a técnica `Log-Sum-Exp`para evitar overflow e underflow de ponto flutuante.
- **Adam Optimizer:** Implementação completa da dinâmica de momentos estocásticos e correção de viés inicial.

---

## **Estrutura do Código**

```text
micrograd-numpy/
├── micrograd/
│   ├── engine.py      # Grafo computacional e cálculo de gradientes
│   ├── nn.py          # Camadas (Linear, Sequential)
│   ├── loss.py        # Critérios de perda (CrossEntropy, MSE)
│   └── optim.py       # Algoritmos de descida de gradiente (SGD, Adam)
├── main.py            # Demonstração em classificação não-linear
└── requirements.txt
```

## **Instalação**

1. Clone o repositório:

   ```Bash
   git clone https://github.com/SEU-USUARIO/micrograd-numpy.git
   cd micrograd-numpy
   ```

2. Crie e ative um ambiente virtual:

   ```Bash
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\activate
   ```

3. Instale a única dependência:

   ```Bash
   pip install -r requirements.txt
   ```

## **Exemplo de Uso Rápido**

```Python

import numpy as np
from micrograd import Tensor, Linear, cross_entropy_loss, Adam

# Dados sintéticos
X = Tensor(np.random.randn(32, 4))
y = Tensor(np.eye(2)[np.random.choice(2, 32)])  # One-hot labels

# Modelo
layer = Linear(4, 2)
optimizer = Adam(layer.parameters(), lr=0.01)

# Ciclo de treinamento
logits = layer(X)
loss = cross_entropy_loss(logits, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Perda calculada: {loss.data:.4f}")
```

## **Como Contribuir**

1. Faça um Fork do projeto.
2. Crie uma Branch para sua Feature (`git checkout -b feature/nova-camada`).
3. Commit suas modificações (`git commit -m 'feat: adiciona camada LayerNorm'`).
4. Faça o Push para a Branch (`git push origin feature/nova-camada`).
5. Abra um Pull Request.

## **Licença**

Distribuído sob a licença MIT. Consulte `LICENSE` para obter mais informações.
