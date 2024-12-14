# SynDisNet: Synthetic Disease Network Dataset

**Subtitle:** A Synthetic Dataset for Graph-Based Healthcare Analysis

## Overview

**SynDisNet** is a large, synthetic dataset of patient health records curated to support graph analysis of disease relationships. It models patients, diseases, disease families, and symptoms with probabilistically defined distributions, making it ideal for network-based research, machine learning experiments, and testing graph algorithms in a healthcare context.

## What We Did

1. **Defined Disease Families & Diseases:**  
   We began by selecting 10 disease families (e.g., Cardiovascular, Neurological, Infectious) and assigned several diseases to each family. This hierarchical structure allows for multi-level graph representations.

2. **Modeled Symptoms & Distributions:**  
   For each disease, we identified a set of symptoms, both numerical (e.g., blood pressure, cholesterol) and categorical (e.g., chest pain presence, smoking status). Each symptom was assigned a statistical distribution (Normal, Categorical) to generate realistic yet synthetic data.

3. **Data Generation in Chunks:**  
   To efficiently produce a large dataset (millions of records), we generated data in small, manageable chunks, writing each chunk to a separate CSV file. After generating all chunks, we concatenated them into a single comprehensive CSV. This approach avoids excessive memory usage and makes the process more modular.

4. **Ensured Scalability & Flexibility:**  
   We provided Python scripts to:
   - Generate the dataset in chunks.
   - Combine chunk files into one large dataset.
   - (Optionally) process the data into graph-friendly formats later.

## Features of the Dataset

- **Multi-Level Healthcare Data:**  
  Includes disease families, diseases, and patients with symptom distributions, enabling hierarchical and heterogeneous graphs.

- **Rich Symptom Space:**  
  Numeric features (e.g., blood glucose, BMI) and categorical features (e.g., chest pain, fatigue severity) provide diverse properties for node attributes and edge weighting.

- **Scalable Size:**  
  The dataset can contain millions of rows, making it suitable for large-scale experiments and stress-testing graph algorithms.

- **Synthetic & License-Free:**  
  The dataset is entirely synthetic, avoiding privacy concerns and allowing for a CC0 or similarly open license, meaning researchers and developers can freely use it without restrictions.

## Why It’s Useful

- **Graph Analysis & Network Science:**  
  The data can be naturally represented as a network: diseases connected to symptoms, patients linking multiple diseases and symptoms, and disease families clustering groups of related conditions. This facilitates research in community detection, link prediction, centrality measures, and other network analyses.

- **Benchmarking & Testing:**  
  Ideal for benchmarking graph-based algorithms, machine learning models, and visual analytics tools. Researchers can test hypothesis about disease co-occurrence, symptom similarity clusters, and hierarchical disease taxonomies.

- **Educational & Exploratory Use:**  
  Instructors can use the dataset to teach data science techniques, from data preprocessing to graph construction and analysis. Students and developers can explore large-scale data handling, feature engineering, and network modeling in a safe, no-risk environment.

## Getting Started

1. **Prerequisites:**
   - Python 3.x
   - `pandas`, `numpy` for data manipulation.
2. **Data Generation:**  
   Run the provided Python scripts to generate the dataset in chunks and then merge them into a single large CSV.

3. **Analysis Tools:**  
   Once generated, the dataset can be loaded into graph frameworks (e.g., NetworkX, Neo4j, or graph databases) or used for ML tasks (e.g., clustering, classification) after appropriate transformations.

## License

This dataset is released under **CC BY-SA 4.0**.
