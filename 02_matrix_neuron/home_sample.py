import numpy as np

def forward(W, x, b):
    """
    y = W * x + b
    Basit doğrusal model (tek nöron)
    """
    return np.dot(W, x) + b


if __name__ == "__main__":
    # Ev özellikleri: [metrekare, oda sayısı, merkeze uzaklık(km)]
    # (veriler normalize edilmemiş, sade anlaşılır olsun)
    x = np.array([[120], [3], [8]], dtype=float)

    # Ağırlıklar: her özelliğe verilen önem derecesi
    W = np.array([[0.6, 0.3, -0.2]], dtype=float)

    # Bias: genel fiyat düzeyi (örnek: şehir ortalaması)
    # sistemin “şehrin ortalama ev fiyatı 100 bin TL civarındadır” bilgisini içerdiğini gösterir.
    b = np.array([[100]], dtype=float)

    # Modelin tahmini (çıktı)
    y_pred = forward(W, x, b)
    print(f"Tahmini fiyat: {y_pred.item():.2f} bin TL")


    # Beklenen çıktı:
    # Tahmini fiyat: 142.40 bin TL
