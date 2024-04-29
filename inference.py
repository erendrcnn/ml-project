# -*- coding: utf-8 -*-
import networkx as nx
import random
import math
import matplotlib.pyplot as plt

def create_Big_Graphs():
  num_nodes = 2000
  BigList_2000 =[]
  random_complete = nx.complete_graph(2000)
  for i in range(2000):
    for j in range(i + 1, 2000):
      if random.random() < 0.5:
        random_complete.remove_edge(i, j)
  BigList_2000.append(random_complete)
  #-------------------------------------------------
  random_multigraph = nx.MultiGraph()
  for i in range(num_nodes):
      random_multigraph.add_node(i)
  for i in range(num_nodes):
      for j in range(i + 1, num_nodes):
          if random.random() < 0.5:
              random_multigraph.add_edge(i, j)
  BigList_2000.append(random_multigraph)
  #-------------------------------------------------
  pseudograph = nx.MultiGraph()
  pseudograph.add_nodes_from(range(num_nodes))
  for node in pseudograph.nodes():
      pseudograph.add_edge(node, node)
      random_nodes = random.sample(list(pseudograph.nodes()), k=3)
      pseudograph.add_edges_from([(node, random_node) for random_node in random_nodes])
  BigList_2000.append(pseudograph)
  #-------------------------------------------------
  planar_graph = nx.random_geometric_graph(num_nodes, radius=0.2)
  while not nx.is_connected(planar_graph):
      planar_graph = nx.random_geometric_graph(num_nodes, radius=0.2)
  BigList_2000.append(planar_graph)
  #-------------------------------------------------
  def random_hamiltonian_graph(n):
      graph = nx.random_regular_graph(2, n)
      for i in range(n):
          for j in range(i + 1, n):
              if random.random() < 0.5 and not graph.has_edge(i, j):
                  graph.add_edge(i, j)
      return graph
  random_hamiltonian_graph = random_hamiltonian_graph(num_nodes)
  BigList_2000.append(random_hamiltonian_graph)
  return BigList_2000

def create_main_graphs(num_nodes):
  main_list =[]
  random_regular_3 = nx.random_regular_graph(3, num_nodes)
  random_regular_20 = nx.random_regular_graph(20, num_nodes)
  main_list.append(random_regular_3)
  main_list.append(random_regular_20)
  #-------------------------------------------------
  # Create a random connected graph by generating a random spanning tree
  random_connected_tree = nx.random_tree(num_nodes)
  # Add additional edges to maintain connectivity
  while not nx.is_connected(random_connected_tree):
      node1 = random.choice(list(random_connected_tree.nodes()))
      node2 = random.choice(list(random_connected_tree.nodes()))
      if node1 != node2 and not random_connected_tree.has_edge(node1, node2):
          random_connected_tree.add_edge(node1, node2)
  main_list.append(random_connected_tree)
  #-------------------------------------------------
  def generate_random_connected_cycle_graph(num_nodes):
      if num_nodes < 3:
          raise ValueError("Number of nodes must be at least 3 for a cycle.")
      cycle_graph = nx.cycle_graph(num_nodes)
      # Connect the cycle to form a connected cycle graph
      connected_cycle_graph = nx.connected_watts_strogatz_graph(num_nodes, 2, 0.1)
      return connected_cycle_graph
  random_cycle_graph = generate_random_connected_cycle_graph(num_nodes)
  main_list.append(random_cycle_graph)
  #-------------------------------------------------
  def generate_random_connected_barabasi_albert_graph(num_nodes, m):
      if num_nodes <= m:
          raise ValueError("Number of nodes must be greater than m for a Barabási-Albert graph.")
      ba_graph = nx.barabasi_albert_graph(num_nodes, m)
      while not nx.is_connected(ba_graph):
          # Add edges to connect the graph
          non_edges = list(nx.non_edges(ba_graph))
          edge_to_add = non_edges[0]
          ba_graph.add_edge(*edge_to_add)
      return ba_graph
  m_parameter = 2
  random_connected_ba_graph = generate_random_connected_barabasi_albert_graph(num_nodes, m_parameter)
  main_list.append(random_connected_ba_graph)
  #-------------------------------------------------
  def generate_random_connected_erdos_renyi_graph(num_nodes, probability):
      er_graph = nx.erdos_renyi_graph(num_nodes, probability)
      while not nx.is_connected(er_graph):
          # Add edges to connect the graph
          non_edges = list(nx.non_edges(er_graph))
          edge_to_add = non_edges[0]
          er_graph.add_edge(*edge_to_add)
      return er_graph
  edge_probability = 0.1  # Adjust as needed
  random_connected_er_graph = generate_random_connected_erdos_renyi_graph(num_nodes, edge_probability)
  main_list.append(random_connected_er_graph)
  #-------------------------------------------------
  k = 2  # Each node is connected to k nearest neighbors
  p = 0.3  # Probability of rewiring each edge
  random_graph_ws = nx.watts_strogatz_graph(num_nodes, k, p)
  main_list.append(random_graph_ws)
  #-------------------------------------------------
  k = 10  # Each node is connected to k nearest neighbors
  p = 0.3  # Probability of rewiring each edge
  random_graph_ws = nx.watts_strogatz_graph(num_nodes, k, p)
  main_list.append(random_graph_ws)
  #-------------------------------------------------

  def generate_connected_random_geometric_graph(num_nodes, radius):
      random_geo_graph = nx.random_geometric_graph(num_nodes, radius)
      while not nx.is_connected(random_geo_graph):
          # Connect the graph by adding edges
          non_edges = list(nx.non_edges(random_geo_graph))
          random_edge = random.choice(non_edges)
          random_geo_graph.add_edge(*random_edge)
      return random_geo_graph
  radius = 0.1  # Adjust as needed
  connected_random_geo_graph = generate_connected_random_geometric_graph(num_nodes, radius)
  main_list.append(connected_random_geo_graph)
  #-------------------------------------------------
  def generate_connected_random_internet_graph(num_nodes, k, p):
      internet_graph = nx.random_internet_as_graph(num_nodes)
      while not nx.is_connected(internet_graph):
          non_edges = list(nx.non_edges(internet_graph))
          random_edge = non_edges[0]
          internet_graph.add_edge(*random_edge)
      undirected_internet_graph = internet_graph.to_undirected()
      connected_random_internet_graph = nx.connected_watts_strogatz_graph(num_nodes, k, p)
      return connected_random_internet_graph
  k_parameter = 5
  p_parameter = 0.5
  connected_random_internet_graph = generate_connected_random_internet_graph(num_nodes, k_parameter, p_parameter)
  main_list.append(connected_random_internet_graph)
  #-------------------------------------------------
  return main_list

BigGraphs = create_Big_Graphs()
main_list = create_main_graphs(5000)
main_list2 = create_main_graphs(8000)

def total_dimensions(graphs):
    total_dims = 0
    for graph in graphs:
        total_dims += len(graph.nodes())
    return total_dims

# Calculate total dimensions for each list of graphs
total_dims_BigGraphs = total_dimensions(BigGraphs)
total_dims_main_list = total_dimensions(main_list)
total_dims_main_list2 = total_dimensions(main_list2)

import pickle
import numpy as np
import networkx as nx

# GCN yayılım fonksiyonu.
def gcn_propagate(adj_matrix, node_embeddings):
    normalized_adj = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
    updated_embeddings = np.dot(normalized_adj, node_embeddings)
    return updated_embeddings

def generate_embeddings(graphs, target_dim=47616, num_iterations=10):
    embeddings = []
    for graph in graphs:
        adj_matrix = nx.to_numpy_array(graph)
        initial_dim = adj_matrix.shape[0]
        node_embeddings = np.random.randn(initial_dim, initial_dim)

        # GCN yayılımını uygula
        for _ in range(num_iterations):
            node_embeddings = gcn_propagate(adj_matrix, node_embeddings)

        # Her bir düğüm için embedding vektörünün ortalamasını al
        graph_embedding = np.mean(node_embeddings, axis=0)

        # Boyut ayarlaması
        if graph_embedding.shape[0] < target_dim:
            padding = np.zeros(target_dim - graph_embedding.shape[0])
            graph_embedding = np.concatenate([graph_embedding, padding])
        elif graph_embedding.shape[0] > target_dim:
            graph_embedding = graph_embedding[:target_dim]

        embeddings.append(graph_embedding)
    return embeddings

# Önceden eğitilmiş bir sınıflandırıcı modeli yükler.
with open('classifier.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

big_graphs_embeddings = generate_embeddings(BigGraphs)
prediction_big = loaded_model.predict(big_graphs_embeddings)
print("Prediction for BigGraphs: ", prediction_big)

# Her bir graflistesi için GCN yayılımını uygular ve tahminler yapar.
main_list_embeddings = generate_embeddings(main_list)
prediction_main = loaded_model.predict(main_list_embeddings)
print("Prediction for main_list: ", prediction_main)

main_list2_embeddings = generate_embeddings(main_list2)
prediction_main2 = loaded_model.predict(main_list2_embeddings)
print("Prediction for main_list2: ", prediction_main2)