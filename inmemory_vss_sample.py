from sklearn.neighbors import NearestNeighbors
import numpy as np

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
