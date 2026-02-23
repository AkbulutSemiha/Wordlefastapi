# Wordle Academic Solver API 🧩

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

A comprehensive research-focused framework for solving Wordle puzzles (specialized for Turkish language) using multiple algorithmic approaches: Rule-Based, Information Theory (Entropy), and Machine Learning (LSTM).

## 🌟 Key Features

- **Multiple Solver Engines**: Comparison between heuristic, probabilistic, and deep learning methods.
- **FastAPI Integration**: Lightweight REST API for real-time guess predictions.
- **Benchmarking Suite**: Automated simulation of thousands of games to evaluate solver performance.
- **Visualization & Analytics**: Tools for generating success rate plots, heatmaps, and statistical comparisons.
- **Academic Reproducibility**: Built-in deterministic seeding for consistent research results.

## 🏗️ Project Structure

```text
├── main.py                 # FastAPI Entry Point
├── run_benchmarks.py       # Simulation & Benchmarking Tool
├── src/
│   ├── core/               # Game logic & Experiment management
│   ├── solvers/            # Implementation of Rule-Based, Entropy, and LSTM solvers
│   ├── infra/              # Data access (Word repositories)
│   └── models/             # PyTorch LSTM model definitions & weights
├── Words/                  # Turkish and English word lists & frequencies
├── Analysis of Result/     # Visualization and statistical analysis scripts
├── DatasetPrepare/         # Scripts for synthetic game data generation
└── ModelPrepare/           # Training notebooks and scripts
```

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the API

Start the FastAPI server for real-time predictions:

```bash
python main.py
```
The documentation will be available at `http://127.0.0.1:8000/docs`.

### 3. Running Benchmarks

Evaluate solver performance across a set of target words:

```bash
python run_benchmarks.py
```

## 🧠 Solver Methodologies

| Solver | Description | Performance |
| :--- | :--- | :--- |
| **Rule-Based** | Heuristic filtering of the word pool based on feedback. | Baseline |
| **Entropy** | Information Theory approach maximizing the expected entropy of each guess. | High Accuracy (Slow) |
| **Hybrid LSTM** | Deep Learning model combined with heuristic filtering for dynamic prediction. | Faster than Entropy |

## 📊 Analysis and Visualization

After running benchmarks, results are saved to `results.csv` and `benchmark_details.csv`. You can generate visualizations using the scripts in `Analysis of Result/`:

```bash
# Example: Generate success rate and step count plots
cd "Analysis of Result"
python plot_result.py
```

## 🔬 Scientific Reproducibility

To ensure fair comparison in academic contexts, all benchmarks utilize a fixed `RUN_SEED = 42`. This ensures that every solver starts with the same "first guess" for a specific game index, eliminating variance caused by random initialization.

---
*Created for Academic Research on Intelligent Wordle Solvers.*
