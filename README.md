# Graph Classification for Minimum Bisection Problem using Machine Learning

## Overview

This project focuses on using machine learning techniques to solve the **Minimum Bisection Problem** on graphs. The Minimum Bisection Problem involves partitioning the vertices of a graph into two equal-sized subsets such that the number of edges between the subsets is minimized. This problem has applications in fields like network optimization, VLSI design, and data clustering.

In this work, we employ machine learning models to classify and analyze graph structures, leveraging their features to predict optimal or near-optimal bisections.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Graph Representation:** Converts input graphs into feature matrices using graph embeddings.
- **Machine Learning Models:** Trains classification models to predict bisection quality.
- **Evaluation Metrics:** Provides metrics like accuracy, F1 score, and edge-cut minimization ratio.
- **Visualization Tools:** Visualizes graph bisection results for analysis.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/erendrcnn/ml-project.git
   cd ml-project
   ```
2. **Set up environment:**
   Ensure Python 3.8+ is installed.

---

## Usage

1. **Prepare the dataset:** Place your graph data in the `data/` directory.
2. **Run Graph Generation:**
   ```bash
   graph_generation.ipynb
   ```
3. **Run Inference:**
   ```bash
   python inference.py
   ```

---

## Dataset

The dataset consists of synthetic and real-world graphs. Each graph includes:

- Nodes and edges
- Edge weights (optional)
- Ground truth for minimum bisection (if available)

Example datasets:

- [Zachary's Karate Club Graph](https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html)
- Custom synthetic graphs generated using the Erdős–Rényi model.

---

## Model Details

### Graph Representation

- **Node Embeddings:** Generated using Node2Vec, GraphSAGE, or Graph Convolutional Networks (GCNs).
- **Graph Features:** Includes degree distributions, clustering coefficients, and adjacency matrix statistics.

### Machine Learning Models

- **Support Vector Machines (SVM):** Used for binary classification with high-dimensional feature handling.
- **Random Forest Classifier:** An ensemble learning approach that combines predictions from multiple decision trees.
- **Perceptron Learning Algorithm (PLA):** A simple and robust linear classifier.

---

## Results

- **Performance Metrics:**
  - Accuracy: 85%
  - F1 Score: 0.83
  - Average Edge-Cut Minimization: 92%
- **Visualization Example:** See `results/` folder for bisection visualizations.

---

## Repository Structure

The repository includes the following files and directories:

### Directories

- `.ipynb_checkpoints/`: Checkpoints for Jupyter Notebook files.
- `Photos/`: Images used for visualization.

### Files

- `.gitignore`: Specifies intentionally untracked files to ignore.
- `Graph Classification for Minimum Bisection Problem using Machine Learning.pdf`: Detailed project report.
- `LICENSE`: License file.
- `README.md`: Project documentation.
- `calc_graph.py`: Script for calculating graph metrics and features.
- `gen_dataset.py`: Script for generating graph datasets.
- `graph.ipynb`: Jupyter Notebook for graph-based experimentation.
- `graph_generation.ipynb`: Notebook for generating graph datasets using the Erdős–Rényi model.
- `graph_generation_old.ipynb`: Older version of the graph generation script.
- `inference.ipynb`: Notebook for running inference on trained models.
- `inference.py`: Python script for inference tasks.
- `inference2.ipynb`: Additional notebook for inference experimentation.
- `inference_script.py`: Python script automating the inference process.
- `random_forest_classifier.py`: Implementation of the Random Forest Classifier for graph embeddings.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a Pull Request.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

