import numpy as np

def forward(W: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Tek katmanlı bir nöronun ileri geçişini (forward pass) hesaplar.
    Denklem: y = W * x + b
    """
    return np.dot(W, x) + b

def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU (Rectified Linear Unit) aktivasyon fonksiyonu.
    Negatif değerleri sıfırlar, pozitifleri olduğu gibi bırakır.
    """
    return np.maximum(0, x)

if __name__ == "__main__":
    # Örnek: 2 girişli, 2 çıkışlı basit bir nöron
    W = np.array([[0.2, 0.8],
                  [0.5, -0.3]], dtype=float)

    x = np.array([[1.0],
                  [2.0]], dtype=float)

    b = np.array([[0.1],
                  [0.2]], dtype=float)

    # 1. Aşama: Doğrusal dönüşüm (Wx + b)
    y_linear = forward(W, x, b)
    print("Doğrusal çıktı (Wx + b):")
    print(y_linear)

    # 2. Aşama: Aktivasyon fonksiyonu (ReLU)
    y_relu = relu(y_linear)
    print("ReLU sonrası çıktı:")
    print(y_relu)
