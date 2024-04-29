from node2vec import Node2Vec
from sklearn.decomposition import PCA
import networkx as nx
import gen_dataset as gd

def calculate_basic_graph_features(graphs):
    """
    Calculate basic graph features for each graph in the list.
    
    Parameters:
        graphs (list): A list of NetworkX graphs.
    
    Returns:
        dict: A dictionary with graph as key and a dictionary of features as value.
    """
    features = {}
    for G in graphs:
        num_nodes = len(G.nodes)
        degrees = np.array([degree for _, degree in G.degree()])
        clustering_coeffs = np.array(list(nx.clustering(G).values()))
        diameter = nx.diameter(G) if nx.is_connected(G) else -1  # -1 indicates disconnected graph
        avg_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else -1

        features[G] = {
            'num_nodes': num_nodes,
            'degree_distribution': degrees,
            'clustering_coefficients': clustering_coeffs,
            'diameter': diameter,
            'average_path_length': avg_path_length
        }
    return features

def generate_node_embeddings(graphs, dimensions=64):
    """
    Generate node embeddings for each graph using Node2Vec.
    
    Parameters:
        graphs (list): A list of NetworkX graphs.
        dimensions (int): Dimensionality of the node embeddings.
    
    Returns:
        dict: A dictionary with graph as key and the node embeddings as value.
    """
    embeddings = {}
    for G in graphs:
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=30, num_walks=200, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings[G] = model.wv.vectors
    return embeddings

def create_feature_matrix(graph_features, graph_embeddings):
    """
    Create a feature matrix by combining graph features and embeddings.
    
    Parameters:
        graph_features (dict): A dictionary of graph features.
        graph_embeddings (dict): A dictionary of graph embeddings.
    
    Returns:
        np.array: A feature matrix.
    """
    feature_matrix = []
    for G in graph_features:
        basic_features = [
            graph_features[G]['num_nodes'],
            np.mean(graph_features[G]['degree_distribution']),
            np.mean(graph_features[G]['clustering_coefficients']),
            graph_features[G]['diameter'],
            graph_features[G]['average_path_length']
        ]
        
        # Flatten node embeddings and concatenate with basic features
        flattened_embeddings = graph_embeddings[G].flatten()
        combined_features = np.concatenate((basic_features, flattened_embeddings))
        
        # Use PCA to reduce dimensions if necessary
        pca = PCA(n_components=100)  # Adjust the number of components as needed
        reduced_features = pca.fit_transform(combined_features.reshape(1, -1))
        
        feature_matrix.append(reduced_features[0])
    
    return np.array(feature_matrix)

random_graphs = gd.generate_random_graphs(10, 50, 100)

# Calculate basic graph features
basic_graph_features = calculate_basic_graph_features(random_graphs)

# Generate node embeddings
node_embeddings = generate_node_embeddings(random_graphs)

# Create feature matrix
feature_matrix = create_feature_matrix(basic_graph_features, node_embeddings)

# Display the shape of the feature matrix
feature_matrix.shape