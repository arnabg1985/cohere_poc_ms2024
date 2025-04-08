
from sklearn.neighbors import NearestNeighbors
import numpy as np
import json

def find_nearest_text_chunks(json_array, query_vector, n_neighbors=2):
    """
    Find the nearest text chunks based on their embeddings.

    :param json_array: List of JSON objects with 'text_chunk' and 'embedding' attributes.
    :param query_vector: A 1D NumPy array representing the query vector.
    :param n_neighbors: Number of nearest neighbors to find (default: 2).
    :return: List of dictionaries containing the text chunks and distances to the query vector.
    """
    # Extract embeddings and text chunks from JSON input
    text_chunks = [item['text_chunk'] for item in json_array]
    embeddings = np.array([item['embedding'] for item in json_array])

    # Ensure the query_vector is reshaped correctly
    query_vector = np.array(query_vector).reshape(1, -1)

    # Create and fit the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    nbrs.fit(embeddings)

    # Perform the search
    distances, indices = nbrs.kneighbors(query_vector)

    # Retrieve the nearest neighbors and their distances
    nearest_neighbors = [
        {"text_chunk": text_chunks[i], "distance": distances[0][j]}
        for j, i in enumerate(indices[0])
    ]

    return nearest_neighbors

# Example usage:
json_data = json.dumps([
    {"text_chunk": "This is the first chunk of text.", "embedding": [0.1, 0.2, 0.3]},
    {"text_chunk": "Here is the second chunk of text.", "embedding": [0.4, 0.5, 0.6]},
    {"text_chunk": "The third chunk is right here.", "embedding": [0.7, 0.8, 0.9]},
    {"text_chunk": "Finally, this is the fourth chunk.", "embedding": [0.2, 0.3, 0.1]}
])

query = [0.15, 0.25, 0.35]  # Example query vector
result = find_nearest_text_chunks(json.loads(json_data), query)
print("Nearest text chunks:")
for res in result:
    print(f"Text Chunk: {res['text_chunk']}, Distance: {res['distance']}")





# Example dataset: an array of vectors
data = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [0.2, 0.3, 0.1]
])

# Query vector to search for similar vectors
query_vector = np.array([0.15, 0.25, 0.35]).reshape(1, -1)

# Create and fit the NearestNeighbors model
nbrs = NearestNeighbors(n_neighbors=2, metric='cosine')  # Use cosine similarity
nbrs.fit(data)

# Perform the search
distances, indices = nbrs.kneighbors(query_vector)

# Display results
print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)

# Optional: Retrieve the actual nearest neighbors
nearest_neighbors = data[indices[0]]
print("Nearest neighbors:", nearest_neighbors)
