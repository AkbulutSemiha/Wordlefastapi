import time
import random
import hashlib
import logging
import pandas as pd
from typing import List, Type, Dict, Tuple
from src.solvers.base_solver import BaseSolver
from src.core.game_logic import generate_feedback

# Configure logging for scientific transparency
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentManager:
    """
    Manages running Wordle simulations with strict scientific reproducibility.
    Ensures each game has a deterministic first guess derived from a global seed.
    """
    def __init__(self, target_words: List[str], all_words: List[str], run_seed: int = 42):
        self.target_words = target_words
        self.all_words = all_words
        self.run_seed = run_seed
        self.results = []
        
        logger.info(f"ExperimentManager initialized with RUN_SEED: {self.run_seed}")
        
        # Pre-generate first guesses for each game index to ensure consistency across different solver runs
        self.game_configs = self._generate_game_configs()

    def _generate_game_configs(self) -> List[Dict]:
        """
        Generates deterministic configurations for each game.
        Each game gets its own RNG derived from the RUN_SEED and its index.
        """
        configs = []
        for idx, target in enumerate(self.target_words):
            # Derive a unique, deterministic seed for this specific game index
            seed_str = f"{self.run_seed}_{idx}"
            game_seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
            
            # Use a local RNG to pick the first guess
            game_rng = random.Random(game_seed)
            first_guess = game_rng.choice(self.all_words)
            
            configs.append({
                "index": idx,
                "target": target,
                "first_guess": first_guess,
                "game_seed": game_seed
            })
        return configs

    def run_simulation(self, solver_class: Type[BaseSolver], solver_args: Dict = None, max_steps: int = 6):
        """
        Runs a simulation for a specific solver class.
        Uses the pre-generated first guess for each game to ensure fair and reproducible comparison.
        """
        solver_name = solver_class.__name__
        logger.info(f"Starting simulation for strategy: {solver_name}")
        
        for config in self.game_configs:
            target = config["target"]
            first_guess = config["first_guess"]
            
            solver = solver_class(self.all_words, **(solver_args or {}))
            history = []
            steps = 0
            success = False
            start_time = time.time()
            
            while steps < max_steps:
                steps += 1
                
                # Deterministic first guess selection
                if steps == 1:
                    guess = first_guess
                else:
                    guess = solver.predict(history)
                
                if not guess:
                    break
                
                feedback = generate_feedback(target, guess)
                history.append((guess, feedback))
                
                if guess == target:
                    success = True
                    break
            
            duration = time.time() - start_time
            self.results.append({
                "Strategy": solver_name,
                "GameIndex": config["index"],
                "Target": target,
                "FirstGuess": first_guess,
                "Steps": steps,
                "Success": success,
                "Duration": duration,
                "Guesses": " -> ".join([g for g, f in history])
            })

    def get_summary(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.results)
        summary = df.groupby("Strategy").agg({
            "Success": "mean",
            "Steps": "mean",
            "Duration": "mean"
        }).rename(columns={
            "Success": "Accuracy",
            "Steps": "Avg Steps",
            "Duration": "Avg Time (s)"
        })
        return summary

    def save_results(self, filename: str = "experiment_results.csv"):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False, encoding="utf-8")
        logger.info(f"Detailed results saved to {filename}")
        
    def save_summary_to_csv(self, filename: str = "results.csv"):
        summary = self.get_summary()
        summary.to_csv(filename, encoding="utf-8")
        logger.info(f"Summary report saved to {filename}")
