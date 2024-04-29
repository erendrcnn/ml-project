"""
METIN EREN DURUCAN - ZEYNEP CINDEMIR - KEREM AY (GROUP 17)

YAP470 PROJECT - Minimum Bisection Problem
This script is used to make predictions on the test graph.

Usage:
    python inference_script.py
    >> Enter the adjacency matrix of the test graph.

Output:
    A prediction of the input graph's minimum bisection.
"""

import numpy as np
import pickle

# Adjacency matrisini kullanıcı girdisi olarak alır ve numpy dizisine dönüştürür.
adjacency_matrix = input()
adjacency_matrix = np.array(adjacency_matrix)

# GCN yayılım fonksiyonu. Bu fonksiyon, düğüm gömülerini güncellemek için kullanılır.
def gcn_propagate(adj_matrix, node_embeddings, dim=64):
    # Adjacency matrisini normalleştirir.
    normalized_adj = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
    # Normalleştirilmiş adjacency matrisi ile düğüm gömülerini çarparak yeni gömüleri hesaplar.
    updated_embeddings = np.dot(normalized_adj, node_embeddings)
    return updated_embeddings

# Düğüm sayısını ve gömü boyutlarını belirler.
num_nodes = len(adjacency_matrix)
embedding_dim = 64
# Rastgele başlangıç düğüm gömüleri oluşturur.
node_embeddings_gcn = np.random.randn(num_nodes, embedding_dim)

# Belirli sayıda iterasyon için GCN yayılımını uygular.
num_iterations = 10
for _ in range(num_iterations):
    node_embeddings_gcn = gcn_propagate(adjacency_matrix, node_embeddings_gcn)

# Önceden eğitilmiş bir sınıflandırıcı modeli yükler.
with open('classifier.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Yüklenen modeli kullanarak düğüm gömülerine dayalı tahminler yapar.
prediction = loaded_model.predict(node_embeddings_gcn)
print(prediction)