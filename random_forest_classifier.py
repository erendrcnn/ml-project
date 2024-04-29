from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import calc_graph as cg
import gen_dataset as gd

# Generate 100 random graphs with node count between 50 and 100
random_graphs = gd.generate_random_graphs(100, 50, 100)

# Label the graphs based on minimum bisection
graph_labels = gd.label_graphs_by_minimum_bisection(random_graphs)

# Calculate basic graph features
# graph_features = cg.calculate_basic_graph_features(random_graphs)

# Generate node embeddings
# graph_embeddings = cg.generate_node_embeddings(random_graphs, dimensions=64)

# Create a feature matrix by combining graph features and embeddings
# feature_matrix = cg.create_feature_matrix(graph_features, graph_embeddings)

# Öncelikle etiketleri sayısal formata çevirelim (divisible: 1, indivisible: 0)
labels_numeric = [1 if label == 'divisible' else 0 for label in graph_labels.values()]

# Veri setini eğitim ve test setlerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels_numeric, test_size=0.3, random_state=42)

# Rastgele Orman sınıflandırıcı modelini oluşturalım ve eğitelim
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Modeli test seti üzerinde değerlendirelim
y_pred = rf_classifier.predict(X_test)

# Değerlendirme metriklerini hesaplayalım
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

accuracy, precision, recall, f1, roc_auc

# Son olarak, modelin tahminlerini görselleştirelim
plt.figure(figsize=(10, 6))
