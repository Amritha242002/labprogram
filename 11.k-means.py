import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
file_path = 'Mall_Customers.csv'
data = pd.read_csv(file_path)
print("Dataset Preview:")
print(data.head())
columns_to_cluster = ['Annual Income (k$)', 'Spending Score(1-100)']
data_subset = data[columns_to_cluster]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_subset)
k = int(input("Enter a K value : "))
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_data)
data['Cluster'] = kmeans.labels_
output_file = 'kmeans_output3.csv'
data.to_csv(output_file, index=False)
print(f"Clustering results saved to {output_file}")
customcmap = ListedColormap(["DeepPink", "Gold", "YellowGreen","OrangeRed", "Cyan"])
fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=kmeans.labels_, cmap=customcmap)
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], 
            	color='DarkBlue', marker='8', s=100,label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel(columns_to_cluster[0])
plt.ylabel(columns_to_cluster[1])
plt.legend()
plt.show()

