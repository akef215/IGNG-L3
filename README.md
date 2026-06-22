# Incremental Growing Neural Gas (IGNG)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

An implementation of the **Incremental Growing Neural Gas (IGNG)** algorithm in PyTorch for **online clustering**, **topology learning**, and **high-dimensional data representation**.

Unlike traditional clustering algorithms such as K-Means, IGNG incrementally constructs a graph that adapts to the underlying structure of the data. This makes it particularly suitable for streaming environments, non-stationary data, and applications where retraining from scratch is impractical.

---

## Overview

Growing Neural Gas (GNG) is a self-organizing neural network introduced by Bernd Fritzke for topology-preserving vector quantization. The Incremental Growing Neural Gas (IGNG) extends this idea by allowing continuous adaptation to incoming data while maintaining a graph representation of the input space.

The algorithm dynamically:

* Creates new nodes when necessary.
* Adapts node positions to observed samples.
* Maintains neighborhood relationships through graph edges.
* Removes obsolete connections.
* Learns the topology of the data distribution.

The resulting graph can be used for clustering, visualization, representation learning, and exploratory data analysis.

---

## Features

### Learning

* Incremental / online learning
* Unsupervised clustering
* Adaptive graph growth
* Dynamic topology discovery
* Continuous model updates
* No predefined number of clusters

### Graph Management

* Automatic node creation
* Edge aging mechanism
* Edge pruning
* Neighbor adaptation
* Graph topology preservation

### Performance

* PyTorch implementation
* CPU and GPU support
* High-dimensional data support
* Scalable to large datasets

### Visualization

* PCA projection
* t-SNE projection
* UMAP projection
* Graph overlay on projected data
* 2D and 3D visualization
* GEXF export for Gephi

---

## Repository Structure

```text
.
├── igng.py
├── visualization.py
├── benchmarks/
├── notebooks/
├── examples/
├── datasets/
├── images/
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/IGNG.git
cd IGNG
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Linux / MacOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Create a model

```python
from igng import IGNG

model = IGNG(
    input_dim=2,
    sigma=1.0,
    age_max=50
)
```

### Train

```python
model.partial_fit(X)
```

### Predict

```python
labels = model.predict(X)
```

---

## Example: Synthetic Dataset

```python
from sklearn.datasets import make_moons
from igng import IGNG

X, _ = make_moons(
    n_samples=5000,
    noise=0.05
)

model = IGNG(
    input_dim=2,
    sigma=1.0,
    age_max=50
)

model.partial_fit(X)
```

---

## Visualization

### PCA

```python
from visualization import IGNGVisualization

vis = IGNGVisualization(model)

vis.pca(
    data=X,
    save_path="pca.png"
)
```

### t-SNE

```python
vis.tsne(
    data=X,
    save_path="tsne.png"
)
```

### UMAP

```python
vis.umap(
    data=X,
    save_path="umap.png"
)
```

---

## Hyperparameters

| Parameter  | Description                                    |
| ---------- | ---------------------------------------------- |
| sigma      | Similarity threshold controlling node creation |
| eps_b      | Learning rate for the Best Matching Unit       |
| eps_n      | Learning rate for neighboring nodes            |
| age_max    | Maximum edge age before removal                |
| mature_age | Minimum age before node maturity               |
| max_nodes  | Maximum graph size                             |

---

## Benchmarks

The implementation has been tested on several synthetic datasets:

### Geometric Datasets

* Blobs
* Moons
* Circles
* Lines
* Swiss Roll
* S-Curve

### Image Datasets

* MNIST

Example evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Adjusted Rand Index (ARI)

---

## MNIST Experiment

The algorithm was evaluated on the MNIST handwritten digit dataset by learning a graph directly in the original 784-dimensional feature space.

The learned graph can then be:

* Visualized through dimensionality reduction techniques.
* Used for nearest-node classification.
* Studied as a topological representation of digit distributions.

---

## Applications

IGNG can be applied to:

### Machine Learning

* Clustering
* Representation learning
* Data exploration

### Computer Vision

* Image clustering
* Feature space organization
* Unsupervised visual learning

### Streaming Data

* Online learning
* Sensor data analysis
* Adaptive systems

### Robotics

* Environment mapping
* Topological learning
* Autonomous navigation

---

## Exporting Graphs

The learned graph can be exported to Gephi-compatible format:

```python
vis.export_gexf("graph.gexf")
```

This enables advanced graph analysis and visualization.

---

## Research Context

This project was developed during a research internship at the **LITIS Laboratory (Laboratoire d'Informatique, de Traitement de l'Information et des Systèmes)** at the University of Rouen Normandy.

The objective of the internship was to study the transition from the classical Growing Neural Gas (GNG) to an Incremental Growing Neural Gas (IGNG) framework for image analysis, clustering, and future applications involving streaming visual data.

---

## Future Work

* Semi-supervised learning
* Adaptive parameter tuning
* Video stream processing
* Real-time computer vision applications
* Integration with deep feature extractors
* Hybrid graph-neural architectures

---

## References

### Growing Neural Gas (GNG)

Fritzke, B. (1995).

**A Growing Neural Gas Network Learns Topologies.**

Advances in Neural Information Processing Systems (NIPS).

```bibtex
@inproceedings{fritzke1995gng,
  title={A Growing Neural Gas Network Learns Topologies},
  author={Fritzke, Bernd},
  booktitle={Advances in Neural Information Processing Systems},
  year={1995}
}
```

### Incremental Growing Neural Gas (IGNG)

Prudent, Y., Ennaji, A., Legrand, P., and Heroux, P.

**Incremental Growing Neural Gas for Online Unsupervised Learning.**

```bibtex
@article{prudent_igng,
  title={Incremental Growing Neural Gas for Online Unsupervised Learning},
  author={Prudent, Yann and Ennaji, Aziz and Legrand, Pierrick and Heroux, Pierre},
  year={2025}
}
```

---

## Author

**Mohamed Elakef Zenagui**

Data Science Student — University of Rouen Normandy

Research Intern at LITIS Laboratory

Interests:

* Machine Learning
* Unsupervised Learning
* Computer Vision
* Graph-based Learning
* Incremental Learning
* Data Mining
