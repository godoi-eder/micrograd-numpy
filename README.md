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
