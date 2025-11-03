# Vektörler ve Kosinüs Benzerliği

Bu proje, iki vektör arasındaki kosinüs benzerliğini hesaplayan bir Python uygulamasını içermektedir.

## Matematiksel Temel

### Kosinüs Benzerliği Nedir?

Kosinüs benzerliği, iki vektör arasındaki açının kosinüsünü ölçerek bu vektörlerin ne kadar benzer olduğunu belirleyen bir metriktir. Bu metrik, vektörlerin yönüne odaklanır ve büyüklüklerini göz ardı eder.

### Matematiksel Formül

İki vektör **A** ve **B** için kosinüs benzerliği şu formülle hesaplanır:

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

Burada:
- **A · B**: İki vektörün nokta çarpımı (dot product)
- **||A||**: A vektörünün büyüklüğü (magnitude/norm)
- **||B||**: B vektörünün büyüklüğü (magnitude/norm)
- **θ**: İki vektör arasındaki açı

### Detaylı Hesaplama Adımları

#### 1. Nokta Çarpımı (Dot Product)
İki vektör A = [a₁, a₂, ..., aₙ] ve B = [b₁, b₂, ..., bₙ] için:

```
A · B = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ = Σᵢ(aᵢ × bᵢ)
```

#### 2. Vektör Büyüklüğü (Euclidean Norm)
Bir vektör A = [a₁, a₂, ..., aₙ] için büyüklük:

```
||A|| = √(a₁² + a₂² + ... + aₙ²) = √(Σᵢ aᵢ²)
```

#### 3. Kosinüs Benzerliği Hesaplama
```
similarity = (A · B) / (||A|| × ||B||)
```

### Sonuç Değerleri

- **1.0**: Vektörler tamamen aynı yönde (θ = 0°)
- **0.0**: Vektörler birbirine dik (θ = 90°)
- **-1.0**: Vektörler tamamen zıt yönde (θ = 180°)
- **(-1, 1) arasındaki değerler**: Vektörler arasındaki açıya bağlı benzerlik

## Kod Açıklaması

### `cosine_similarity` Fonksiyonu

```python
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
```

Bu fonksiyon şu adımları gerçekleştirir:

1. **Nokta Çarpımı Hesaplama**: `np.dot(vec_a, vec_b)`
2. **Büyüklük Hesaplama**: `np.linalg.norm()` kullanarak her vektörün büyüklüğünü hesaplar
3. **Sıfır Kontrol**: Herhangi bir vektörün büyüklüğü sıfırsa 0.0 döndürür
4. **Kosinüs Hesaplama**: Nokta çarpımını büyüklüklerin çarpımına böler

### Örnek Hesaplama

Kod örneğinde:
- **vector1 = [1, 2, 3]**
- **vector2 = [4, 5, 6]**

**Adım 1 - Nokta Çarpımı:**
```
1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

**Adım 2 - Büyüklükler:**
```
||vector1|| = √(1² + 2² + 3²) = √(1 + 4 + 9) = √14 ≈ 3.742
||vector2|| = √(4² + 5² + 6²) = √(16 + 25 + 36) = √77 ≈ 8.775
```

**Adım 3 - Kosinüs Benzerliği:**
```
similarity = 32 / (3.742 × 8.775) ≈ 32 / 32.832 ≈ 0.975
```

## Kullanım Alanları

Kosinüs benzerliği şu alanlarda yaygın olarak kullanılır:

- **Metin Madenciliği**: Dökümanlar arası benzerlik
- **Makine Öğrenmesi**: Özellik vektörleri karşılaştırması
- **Öneri Sistemleri**: Kullanıcı veya ürün benzerliği
- **Bilgisayarlı Görü**: Görüntü özellik vektörleri
- **Doğal Dil İşleme**: Kelime vektörleri (word embeddings)

## Kurulum ve Çalıştırma

### Gereksinimler
```bash
pip install numpy
```

### Çalıştırma
```bash
python main.py
```

## Avantajlar

1. **Büyüklük Bağımsız**: Vektör büyüklüklerinden etkilenmez
2. **Normalleştirilmiş**: Sonuç her zaman [-1, 1] aralığında
3. **Hesaplama Verimliliği**: Nispeten hızlı hesaplanabilir
4. **Geometrik Anlam**: Açı kavramıyla doğrudan ilişkili

## Not

Bu implementasyon, sıfır vektörler için güvenlik kontrolü içermektedir. Eğer herhangi bir vektörün büyüklüğü sıfırsa, fonksiyon 0.0 değeri döndürür.