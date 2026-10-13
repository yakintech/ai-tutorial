import numpy as np


def forward(W, X, b):
    """
    Y = W * X + b
    Tek ev için de, yüzlerce ev için de aynı fonksiyon.
    """
    return np.dot(W, X) + b


def relu(x):
    return np.maximum(0, x)


if __name__ == "__main__":
    # Ağırlıklar ve bias, home_sample.py ile birebir aynı
    W = np.array([[0.6, 0.3, -0.2]], dtype=float)   # (1, 3)
    b = np.array([[100]], dtype=float)              # (1, 1)

    # Tek fark: x artık bir sütun değil, bir tablo (matris).
    # Her SÜTUN bir ev: [metrekare, oda sayısı, uzaklık]
    #            ev1   ev2   ev3   ev4
    X = np.array([[120,   80,  200,   60],   # metrekare
                  [  3,    2,    5,    1],   # oda sayısı
                  [  8,    2,   30,  450]], dtype=float)  # uzaklık (km)
    # X: (3, 4) -> 3 özellik, 4 ev

    Y = relu(forward(W, X, b))   # (1, 3) x (3, 4) + (1, 1) -> (1, 4)

    for i, fiyat in enumerate(Y[0], start=1):
        print(f"ev{i}: {fiyat:.2f} bin")

    # Beklenen çıktı:
    # ev1: 171.30 bin
    # ev2: 148.20 bin
    # ev3: 215.50 bin
    # ev4: 46.30 bin   (0.6*60 + 0.3*1 - 0.2*450 + 100 = 46.30)
    #
    # 4 ev yerine 500 ev olsaydı: X'e 496 sütun daha eklenirdi, kod değişmezdi.
    # Ekran kartlarının işi de tam olarak bu: tek bir çarp-topla, binlerce kez, aynı anda.
