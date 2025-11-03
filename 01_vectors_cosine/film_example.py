import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Film özellik vektörleri
# [aksiyon seviyesi, romantizm oranı, komedi yoğunluğu]

movies = {
    "Matrix": np.array([9, 1, 2]),   # bol aksiyon, az romantizm, az komedi
    "Titanic": np.array([2, 9, 3]),  # çok romantizm, az aksiyon, az komedi
    "Deadpool": np.array([8, 3, 9])  # aksiyon + komedi
}

query = movies["Matrix"]

for name, vector in movies.items():
    sim = cosine_similarity(query, vector)
    print(f"{name} benzerlik: {sim:.2f}")

# Beklenen çıktı:
# Matrix benzerlik: 1.00
# Titanic benzerlik: düşük değer (örneğin 0.3 civarı)
# Deadpool benzerlik: orta değer (örneğin 0.7 civarı

