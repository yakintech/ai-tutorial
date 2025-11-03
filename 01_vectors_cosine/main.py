import numpy as np 


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculate the cosine similarity between two vectors.

    Args:
        vec_a (np.ndarray): First vector.
        vec_b (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity between vec_a and vec_b.
    """
    dot_product: float = np.dot(vec_a, vec_b)

    magnitude_a: float = np.linalg.norm(vec_a)
    magnitude_b: float = np.linalg.norm(vec_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


if __name__ == "__main__":
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])

    similarity = cosine_similarity(vector1, vector2)
    print(f"Cosine Similarity: {similarity}")

    
