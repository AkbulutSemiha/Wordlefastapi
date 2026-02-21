import os
from src.infra.word_repository import WordRepository
from src.core.experiment import ExperimentManager
from src.solvers.rule_based_solver import RuleBasedSolver
from src.solvers.entropy_solver import EntropySolver
from src.solvers.hybrid_lstm_solver import HybridLSTMSolver

def run_benchmarks():
    # 0. Scientific Reproducibility
    RUN_SEED = 42 # Constant seed for academic publication reproducibility
    
    # 1. Initialize Repository
    repo = WordRepository(data_dir="Words")
    all_words = repo.get_words(language="tr")
    # For simulation, use a subset of frequent words as targets
    target_words = repo.get_frequent_words(language="tr", limit=10) # Using 10 words for a meaningful benchmark
    
    # 2. Initialize Experiment Manager with the global RUN_SEED
    manager = ExperimentManager(target_words, all_words, run_seed=RUN_SEED)
    
    # 3. Run Experiments
    
    # Baseline: Rule-Based
    manager.run_simulation(RuleBasedSolver)
    
    # Information Theory: Entropy
    # Warning: This is slow for large pools
    manager.run_simulation(EntropySolver)
    
    # Research Focus: Hybrid LSTM
    model_path = "src/models/tr_LSTMmodel_100epoch.pth"
    if os.path.exists(model_path):
        manager.run_simulation(HybridLSTMSolver, solver_args={"model_path": model_path})
    else:
        print(f"Skipping Hybrid LSTM: Model not found at {model_path}")

    # 4. Reporting
    print("\nBenchmark Summary:")
    print(manager.get_summary())
    
    manager.save_results("benchmark_details.csv")
    manager.save_summary_to_csv("results.csv")

if __name__ == "__main__":
    run_benchmarks()
