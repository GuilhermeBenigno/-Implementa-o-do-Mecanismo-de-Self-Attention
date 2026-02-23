import numpy as np


class ScaledDotProductAttention:
    def __init__(self):
        pass

    @staticmethod
    def softmax(x):
        
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, Q, K, V):

        d_k = K.shape[1]

        # Produto escalar QK^T
        scores = np.matmul(Q, K.T)

        # Aplicação do Scaling Factor
        scaled_scores = scores / np.sqrt(d_k)

        # Softmax linha a linha
        attention_weights = self.softmax(scaled_scores)

        # Multiplicação pelos Values
        output = np.matmul(attention_weights, V)

        return output, attention_weights
