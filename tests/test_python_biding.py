# import numpy as np
# import faiss

# # Generate some random data
# d = 64  # dimension
# n = 10000  # number of data points
# np.random.seed(123)
# xb = np.random.random((n, d)).astype('float32')

# # Build the index
# index = faiss.IndexFlatL2(d)
# index.add(xb)

# # Set up the query vector
# xq = np.random.random((1, d)).astype('float32')

# # Set the radius for the range search
# radius = 0.2

# # Perform the range search
# D, I = index.boundary_search(xq, 0.2, 0.5)

# # Print the results
# print("Indices of neighbors within radius:", I)
# print("Distances to neighbors within radius:", D)
