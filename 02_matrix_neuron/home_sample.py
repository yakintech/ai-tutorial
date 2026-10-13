import numpy as np


def forward(W, x, b):
    """
    y = W * x + b
    Basit doğrusal model (tek nöron)
    """
    return np.dot(W, x) + b


def relu(x):
    """
    Eşik: negatifse sıfır, pozitifse olduğu gibi kalsın.
    """
    return np.maximum(0, x)


if __name__ == "__main__":
    # Ev özellikleri: [metrekare, oda sayısı, merkeze uzaklık (km)]
    # (veriler normalize edilmemiş, sade anlaşılır olsun)
    x = np.array([[120], [3], [8]], dtype=float)

    # Ağırlıklar: her özelliğe verilen önem derecesi
    # Uzaklığın ağırlığı EKSİ: uzaklık arttıkça değer düşüyor.
    W = np.array([[0.6, 0.3, -0.2]], dtype=float)

    # Bias: genel fiyat düzeyi.
    # "Şehrin ortalama ev fiyatı 100 bin civarındadır" bilgisini taşır.
    b = np.array([[100]], dtype=float)

    # 1) Ağırlıklı toplam + bias
    y_linear = forward(W, x, b)
    print(f"Doğrusal çıktı (Wx + b): {y_linear.item():.2f} bin")
    # 0.6*120 + 0.3*3 + (-0.2)*8 + 100 = 72 + 0.9 - 1.6 + 100 = 171.30

    # 2) Eşik (ReLU)
    y = relu(y_linear)
    print(f"ReLU sonrası karar     : {y.item():.2f} bin")
    # Pozitif olduğu için değişmedi: 171.30

    print()

    # Aynı ev, ama merkeze 900 km uzakta olsaydı?
    x_uzak = np.array([[120], [3], [900]], dtype=float)
    y_linear_uzak = forward(W, x_uzak, b)
    y_uzak = relu(y_linear_uzak)
    print(f"Uzak ev doğrusal çıktı : {y_linear_uzak.item():.2f} bin")
    # 72 + 0.9 - 180 + 100 = -7.10  -> eksi fiyat olmaz
    print(f"Uzak ev ReLU sonrası   : {y_uzak.item():.2f} bin")
    # ReLU negatifi sıfıra çekti: "ilgilenmiyorsan hiç konuşma"
