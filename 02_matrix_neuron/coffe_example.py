import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU (Rectified Linear Unit) aktivasyon fonksiyonu.
    Negatif değerleri 0 yapar, pozitifleri olduğu gibi bırakır.
    """
    return np.maximum(0, x)

def coffee_score(x_0to10: np.ndarray, W: np.ndarray, b: np.ndarray):
    """
    Basit bir nöronun kahve kararı üzerinden simülasyonu.

    Parametreler:
      x_0to10 : (3,1) numpy dizisi
          Girdiler (0–10 arası puanlar): [fiyat, tatlılık, kafein]

      W : (1,3) numpy dizisi
          Ağırlıklar — her özelliğe verilen önem.
          Örnek: fiyat = 0.7, tatlılık = 0.3, kafein = 0.1

      b : (1,1) numpy dizisi
          Bias — genel eğilim.
          Eğer kahveyi genel olarak seviyorsan pozitif olabilir (ör. +0.5)

    Dönüş:
      (float, float)
          (Doğrusal skor Wx+b, ReLU sonrası karar)
    """
    # 0–10 aralığındaki girdileri 0–1 ölçeğine çekelim (normalize)
    x = x_0to10 / 10.0                       # (3,1)
    y_linear = np.dot(W, x) + b              # Ağırlıklı toplam + bias
    y = relu(y_linear)                       # Negatifleri sıfırla
    return y_linear.item(), y.item()

if __name__ == "__main__":
    # Özellikler: [fiyat, tatlılık, kafein] (0–10)
    x = np.array([[3], [5], [9]], dtype=float)

    # Ağırlıklar (önem derecesi): fiyat çok önemli, tatlılık orta, kafein az
    W = np.array([[0.7, 0.3, 0.1]], dtype=float)   # (1,3)

    # Genel eğilim (bias): kahveyi seviyorsan pozitif bias
    b = np.array([[0.5]], dtype=float)             # (1,1)

    # Hesaplama
    linear, decision = coffee_score(x, W, b)

    print(f"Doğrusal skor (Wx+b): {linear:.3f}")
    print(f"ReLU sonrası karar: {decision:.3f}")
