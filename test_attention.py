import numpy as np
from attention import ScaledDotProductAttention


def main():
    # Exemplo simples
    Q = np.array([[1, 0, 1]])
    K = np.array([[1, 0, 1],
                  [0, 1, 0],
                  [1, 1, 0]])
    V = np.array([[1, 0],
                  [0, 1],
                  [1, 1]])

    attention = ScaledDotProductAttention()
    output, weights = attention.forward(Q, K, V)

    print("Pesos de Atenção:")
    print(weights)

    print("\nSaída Final:")
    print(output)


if __name__ == "__main__":
    main()
