import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree

def generate_random_graphs(num_graphs, min_nodes, max_nodes):
    """
    Generate a set of random graphs using the Erdős-Rényi model.
    
    Parameters:
        num_graphs (int): Number of graphs to generate.
        min_nodes (int): Minimum number of nodes in the graphs.
        max_nodes (int): Maximum number of nodes in the graphs.
    
    Returns:
        list: A list of generated NetworkX graphs.
    """
    graphs = []
    for _ in range(num_graphs):
        num_nodes = np.random.randint(min_nodes, max_nodes)
        probability = np.random.uniform(0.1, 0.5)  # Randomly chosen probability
        G = nx.erdos_renyi_graph(num_nodes, probability)
        graphs.append(G)
    return graphs

def label_graphs_by_minimum_bisection(graphs):
    """
    Label the graphs based on whether they can be divided into two equal parts with minimum bisection.
    
    Parameters:
        graphs (list): A list of NetworkX graphs.
    
    Returns:
        dict: A dictionary with graph as key and label ('divisible' or 'indivisible') as value.
    """
    labels = {}
    for G in graphs:
        # Calculate the adjacency matrix
        adjacency_matrix = nx.to_scipy_sparse_matrix(G)
        
        # Find the minimum spanning tree
        mst = minimum_spanning_tree(adjacency_matrix)
        
        # Check if the graph can be divided into two equal parts
        # This is a simplified approach and should be replaced with a more accurate method
        if mst.nnz <= len(G.nodes) / 2:
            labels[G] = 'divisible'
        else:
            labels[G] = 'indivisible'
    
    return labels

# Generate 10 random graphs with node count between 50 and 100
random_graphs = generate_random_graphs(10, 50, 100)

# Label the graphs based on minimum bisection
graph_labels = label_graphs_by_minimum_bisection(random_graphs)

# Display the first graph and its label
nx.draw(random_graphs[0], with_labels=True)
plt.show()
graph_labels[random_graphs[0]]