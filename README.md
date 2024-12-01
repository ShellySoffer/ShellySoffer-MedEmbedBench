
# MedEmbedEval: A Scalable Framework for Evaluating Medical Text Embedding Models

## Overview
**MedEmbedEval** is a benchmarking framework for evaluating text embedding models on semantic tasks in healthcare. This repository includes datasets, code, and tools to assess the performance of various embedding models using real-world, biomedical, and synthetic data.

---

## 📂 Repository Structure

```
|-- README.md
|-- Data/
    |-- pubmed_abstracts_search_queries.xlsx
    |-- pubmed_abstracts_keywords.xlsx
    |-- synthetic_ehr_search_queries.xlsx
|-- Code/
    |-- embeddings_evaluation.py
```

---

## 📊 Dataset Overview

### 1. **PubMed Abstracts to Search Queries**
   - **File Name:** `pubmed_abstracts_search_queries.xlsx`
   - **Description:** Contains pairs of PubMed abstracts and corresponding search queries generated using LLM prompts.
   - **Columns:**
     - `Source Text`: PubMed abstract
     - `Destination Text`: Corresponding search query

### 2. **PubMed Abstracts to Keywords**
   - **File Name:** `pubmed_abstracts_keywords.xlsx`
   - **Description:** Features pairs of PubMed abstracts and their extracted keywords.
   - **Columns:**
     - `Source Text`: PubMed abstract
     - `Destination Text`: Extracted keyword(s)

### 3. **Synthetic EHR Notes to Search Queries**
   - **File Name:** `synthetic_ehr_search_queries.xlsx`
   - **Description:** Synthetic electronic health record (EHR) notes paired with search queries.
   - **Columns:**
     - `Source Text`: Synthetic EHR note
     - `Destination Text`: Corresponding search query

---

## 🛠 Code Description

The `embeddings_evaluation.py` script benchmarks text embedding models on multiple semantic tasks. It uses datasets from real-world and synthetic sources to evaluate embeddings using metrics like cosine similarity.

### Key Features
- **Dynamic Noise Generation**: Creates chimeric descriptions with varying levels of noise to test model robustness.
- **Task-Specific Evaluation**: Benchmarks models on short and long medical text tasks.
- **Memory Optimization**: Includes steps for efficient memory management during GPU processing.

---

## 🚀 Getting Started

### Prerequisites
1. Install Python (>=3.8) and the following dependencies:
   ```bash
   pip install pandas numpy sentence-transformers torch scikit-learn transformers tqdm
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MedEmbedEval.git
   cd MedEmbedEval
   ```

3. Place the datasets in the `Data/` folder.

### Running the Script
1. Navigate to the `Code/` directory:
   ```bash
   cd Code
   ```

2. Execute the script:
   ```bash
   python embeddings_evaluation.py
   ```

3. Outputs:
   - Pickled result files saved in the `Results/` folder.
   - Performance metrics printed to the console.

---

## 🔧 Customization

- **Add/Remove Models**: Update the `models` list in the script to include specific models for evaluation.
- **Adjust Batch Sizes**: Modify the `get_batch_size()` function to tune processing efficiency based on GPU memory.

---

## 📜 Citation

If you use this repository, please cite the accompanying paper:

```
Soffer S, Gendler M, Glicksberg BS, et al. "A Scalable Framework to Benchmark Embedding Models in Semantic Health-Care Tasks."
```

---

## 🤝 Contributions

We welcome contributions to improve this framework or adapt it for additional use cases. Feel free to open an issue or submit a pull request.

---

