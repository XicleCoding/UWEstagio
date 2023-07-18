from sklearn.cluster import KMeans
import numpy as np

complaints = [
    "Loud construction noise near my residence",
    "Potholes on Main Street causing traffic congestion",
    "Concerns about air pollution from nearby factories",
    "Request for additional streetlights in the neighborhood",
    "Excessive noise from a nearby nightclub",
    "Health concerns due to unsanitary conditions in a park",
    "Proposal to build a new community center",
    "Complaint about zoning regulations for a new development",
    "Inquiry about upcoming community events",
    "Report a pothole on Elm Street",
    "Vandalism in the local park",
    "Problems with public transportation",
    "Graffiti on buildings in downtown area",
    "Request for improved recycling facilities",
    "Noise complaint from a busy intersection",
    "Concerns about stray animals in the neighborhood",
    "Complaint about inadequate street cleaning",
    "Water leak in front of my house",
    "Lack of playground equipment in local park",
    "Request for more bike lanes in the city"
]
labels = ["Public Safety", "Infrastructure Issues", "Environmental Concerns", "Traffic and Transportation",
          "Noise and Nuisance", "Public Health", "Parks and Recreation", "Zoning and Land Use",
          "Community Events and Programs", "Civic Services"]

# Convert complaints to numerical representation using a suitable method (e.g., TF-IDF)
# Here, we simply represent each complaint as a bag of words using count vectorization
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
complaints_vectorized = vectorizer.fit_transform(complaints)

# Determine the optimal number of clusters using the elbow method
# and apply k-means clustering
num_clusters = len(labels)  # Set the number of clusters based on the number of labels
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(complaints_vectorized)

# Assign labels to clusters
cluster_labels = kmeans.labels_

# Print the cluster index for each complaint
print("Cluster Index for each Complaint:")
for i, complaint in enumerate(complaints):
    print("Complaint:", complaint)
    print("Cluster Index:", cluster_labels[i])
    print()
