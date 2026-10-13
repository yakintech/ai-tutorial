# Matrix İşlemleri ve Yapay Nöron

Bu proje, yapay sinir ağlarının temel yapı taşı olan nöronların matematik temellerini ve gerçek hayat örneklerini göstermektedir.

## 🧠 Nöron Nedir?

Yapay nöron, beynimizin karar verme sürecini taklit eden matematiksel bir modeldir. İki temel aşamadan oluşur:

1. **Doğrusal Dönüşüm**: Girdileri ağırlıklarla çarpıp toplama
2. **Aktivasyon Fonksiyonu**: Sonucu işleyerek karar verme

## ☕ Gerçek Hayattan Örnek: Kafede Seçim Yapmak

Diyelim ki bir kafedesin. Kahve mi, smoothie mi içeceğine karar vereceksin.
Karar verirken 3 şeye bakıyorsun:

- **Fiyat**
- **Tatlılık seviyesi**  
- **Kafein oranı**

Her birine farklı önem veriyorsun:
- Fiyat senin için çok önemli (ağırlık: 0.7)
- Tatlılık orta derecede önemli (ağırlık: 0.3)
- Kafein az önemli (ağırlık: 0.1)

### 📊 Kahve Puanlama Tablosu

| Özellik  | Değer (0-10) | Ağırlık (Weight) |
|----------|--------------|------------------|
| Fiyat    | 3            | 0.7              |
| Tatlılık | 5            | 0.3              |
| Kafein   | 9            | 0.1              |

### 🧮 Beynin Hesap Yapma Süreci

```
toplam etki = 3×0.7 + 5×0.3 + 9×0.1 = 2.1 + 1.5 + 0.9 = 4.5
```

Bu aslında bir **nokta çarpımı (dot product)** işlemidir.

## 🏠 Gerçek Hayattan Örnek: Ev Almak (`home_sample.py`)

İstanbul'da bir ev bakıyoruz. Üç özellik: **metrekare, oda sayısı, merkeze uzaklık**.

| Özellik          | Değer | Ağırlık |
|------------------|-------|---------|
| Metrekare        | 120   | 0.6     |
| Oda sayısı       | 3     | 0.3     |
| Merkeze uzaklık  | 8 km  | **-0.2**|

Uzaklığın ağırlığı **eksi**: uzaklık arttıkça değer düşüyor. Bir ağırlığın en güçlü tarafı büyüklüğü değil, işareti.

Bias = 100: "şehrin ortalama ev fiyatı 100 bin civarındadır" bilgisi. Özellikler sıfır olsa bile modelin söyleyeceği taban değer.

```
y = 0.6×120 + 0.3×3 + (-0.2)×8 + 100
  = 72 + 0.9 - 1.6 + 100
  = 171.30 bin
```

Aynı ev **900 km** uzakta olsaydı:

```
y = 72 + 0.9 - 180 + 100 = -7.10   →   ReLU   →   0
```

Eksi fiyat olmaz; ReLU negatifi sıfıra çeker. "İlgilenmiyorsan hiç konuşma."

> Not: Bu örnekte veriler kasıtlı olarak normalize edilmemiştir (120, 3, 8 ham değerler). Sade olsun diye. Gerçek modellerde özellikler aynı ölçeğe çekilir; kahve örneğinde bunu yapıyoruz.

## 📦 Tek Karardan Yüzlerce Karara (`batch_example.py`)

Beş yüz ev için fiyat tahmin etmek gerekse ne değişir? **Hiçbir şey.** `x` bir sütun olmaktan çıkıp bir tablo (matris) olur, her sütun bir ev:

```
            ev1  ev2  ev3  ev4
X = [[120,  80, 200,  60],    # metrekare
     [  3,   2,   5,   1],    # oda
     [  8,   2,  30, 450]]    # uzaklık
```

```
Y = W · X + b     (1,3) · (3,4) + (1,1)  →  (1,4)
```

Tek `np.dot` çağrısı, dört fiyat:

```
ev1: 171.30   ev2: 148.20   ev3: 215.50   ev4: 46.30
```

500 ev için `X`'e 496 sütun daha eklenir, kod değişmez. Ekran kartlarının yapay zekayla bu kadar iyi anlaşmasının sebebi bu: her şey tek bir çarp-topla işlemine iniyor ve bu işlem paralel yapılabiliyor.

## 🔢 Matematiksel Temel

### Nöron Formülü

Bir nöronun çıktısı şu adımlarla hesaplanır:

```
y = f(Wx + b)
```

Burada:
- **x**: Giriş vektörü (input vector)
- **W**: Ağırlık matrisi (weight matrix)  
- **b**: Bias (önyargı) vektörü
- **f**: Aktivasyon fonksiyonu
- **y**: Çıktı vektörü

### Detaylı Hesaplama

#### 1. Doğrusal Dönüşüm (Linear Transformation)

```
z = Wx + b
```

Matrix çarpımı şeklinde:
```
[z₁]   [w₁₁ w₁₂ w₁₃] [x₁]   [b₁]
[z₂] = [w₂₁ w₂₂ w₂₃] [x₂] + [b₂]
                       [x₃]
```

#### 2. Aktivasyon Fonksiyonu

**ReLU (Rectified Linear Unit)** fonksiyonu:
```
f(x) = max(0, x)
```

- Negatif değerleri 0 yapar
- Pozitif değerleri olduğu gibi bırakır
- Doğrusal olmayan özellik kazandırır

## 💻 Kod Açıklaması

### `main.py` - Temel Nöron İmplementasyonu

#### `forward` Fonksiyonu
```python
def forward(W: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dot(W, x) + b
```

Bu fonksiyon doğrusal dönüşümü gerçekleştirir:
- Matrix çarpımı: `W × x`
- Bias ekleme: `+ b`

#### `relu` Fonksiyonu
```python
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)
```

ReLU aktivasyon fonksiyonu:
- Negatif değerler → 0
- Pozitif değerler → değişmez

### `home_sample.py` - Ev Örneği

Normalize edilmemiş ham verilerle tek nöron. Önce normal bir ev, sonra çok uzak bir ev ile ReLU'nun negatifi sıfırladığı gösterilir.

### `batch_example.py` - Toplu Hesap

Aynı `W` ve `b` ile 4 evin fiyatını tek matris çarpımında hesaplar.

### `coffe_example.py` - Kahve Örneği

#### Özelleştirilmiş Nöron
```python
def coffee_score(x_0to10: np.ndarray, W: np.ndarray, b: np.ndarray):
```

Bu fonksiyon:
1. Girdileri normalize eder (0-10 → 0-1)
2. Ağırlıklı toplamı hesaplar
3. ReLU aktivasyonu uygular
4. Karar skorunu döndürür

## 📈 Örnek Hesaplama

### Giriş Değerleri
```python
x = [[3], [5], [9]]  # [fiyat, tatlılık, kafein]
W = [[0.7, 0.3, 0.1]]  # ağırlıklar
b = [[0.5]]  # bias
```

### Adım Adım Hesaplama

**1. Normalizasyon:**
```
x_norm = [3/10, 5/10, 9/10] = [0.3, 0.5, 0.9]
```

**2. Doğrusal Dönüşüm:**
```
z = W·x + b = [0.7×0.3 + 0.3×0.5 + 0.1×0.9] + 0.5
z = [0.21 + 0.15 + 0.09] + 0.5 = 0.45 + 0.5 = 0.95
```

**3. ReLU Aktivasyon:**
```
y = max(0, 0.95) = 0.95
```

## 🎯 Sonuç Yorumlama

- **0.95** → Yüksek skor, kahve içmeye değer!
- **0 yakını** → Düşük skor, kahve almayı düşün
- **Negatif → 0** → ReLU sayesinde, kesinlikle almama

## ▶️ Çalıştırma

```bash
pip install numpy
python main.py           # 2 nöronlu küçük katman, ReLU'nun negatifi susturması
python home_sample.py    # ev örneği
python batch_example.py  # 4 ev tek seferde
python coffe_example.py  # kahve örneği (normalize edilmiş girdi)
```

## 🚀 Kullanım Alanları

### Yapay Sinir Ağlarında Nöronlar

1. **Görüntü İşleme**: Piksel değerlerinden özellik çıkarma
2. **Doğal Dil İşleme**: Kelime vektörlerinden anlam çıkarma  
3. **Öneri Sistemleri**: Kullanıcı tercihlerinden öneriler
4. **Oyun AI'ları**: Oyun durumundan optimal hamle seçimi

### Matrix İşlemleri

- **Toplu İşlem (Batch Processing)**: Birden fazla örneği aynı anda işleme
- **Paralelleştirme**: GPU'larda hızlı hesaplama
- **Boyut İndirme/Artırma**: Özellik dönüşümü



## 📊 Matrix Boyutları

### main.py Örneği
```
W: (2, 2) - 2 giriş, 2 çıkış
x: (2, 1) - 2 özellik
b: (2, 1) - 2 bias
y: (2, 1) - 2 sonuç
```

### home_sample.py Örneği
```
W: (1, 3) - 3 giriş, 1 çıkış
x: (3, 1) - 3 özellik
b: (1, 1) - 1 bias
y: (1, 1) - 1 fiyat tahmini
```

### batch_example.py Örneği
```
W: (1, 3) - 3 giriş, 1 çıkış
X: (3, 4) - 3 özellik, 4 ev (her sütun bir ev)
b: (1, 1) - 1 bias (her sütuna yayınlanır / broadcast)
Y: (1, 4) - 4 fiyat tahmini
```

### coffe_example.py Örneği
```
W: (1, 3) - 3 giriş, 1 çıkış  
x: (3, 1) - 3 özellik
b: (1, 1) - 1 bias
y: (1, 1) - 1 karar skoru
```

## 🧭 Neden Matrix Kullanıyoruz?

1. **Verimlilik**: Çok sayıda işlemi tek seferde yapabilir
2. **Ölçeklenebilirlik**: Binlerce nöron aynı anda çalışabilir
3. **GPU Desteği**: Paralel hesaplama imkanı
4. **Matematiksel Zarafet**: Temiz ve anlaşılır formüller

## 💡 Önemli Notlar

- **Bias**: Nöronun genel eğilimini belirler (pozitif = iyimser, negatif = kötümser)
- **Ağırlıklar**: Hangi özelliklerin daha önemli olduğunu belirler
- **Aktivasyon**: Nöronun "ateşlenip ateşlenmeyeceğini" belirler
- **Normalizasyon**: Farklı ölçeklerdeki verileri eşit düzeye getirir

Bu temel kavramlar, derin öğrenmenin (deep learning) yapı taşlarıdır! 🚀
