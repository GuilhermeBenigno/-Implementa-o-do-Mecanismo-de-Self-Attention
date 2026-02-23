# -Implementa-o-do-Mecanismo-de-Self-Attention
# Scaled Dot-Product Attention

Implementação do mecanismo de Scaled Dot-Product Attention conforme descrito no paper "Attention Is All You Need".

## 📌 Como executar

1. Clone o repositório:
   git clone <[link-do-repositorio](https://github.com/GuilhermeBenigno/-Implementa-o-do-Mecanismo-de-Self-Attention/tree/main)>

2. Instale as dependências:
   pip install -r requirements.txt

3. Execute o teste:
   python test_attention.py

---

## 📌 Fórmula Implementada

Attention(Q, K, V) = softmax( (QK^T) / sqrt(d_k) ) V

---

## 📌 Normalização (Scaling Factor)

Após o cálculo do produto escalar QK^T, os valores são divididos por:

sqrt(d_k)

onde:
- d_k = dimensão das chaves (Key)

Isso evita que valores muito grandes prejudiquem o Softmax.

---

## 📌 Exemplo

Input:

Q = [[1, 0, 1]]

Output esperado:
Uma matriz contendo os pesos normalizados e a combinação ponderada de V.

