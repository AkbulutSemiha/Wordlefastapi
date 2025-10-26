"""
DatasetPrepare Package
======================

This package is designed to generate Wordle-style datasets for AI model training.
It simulates human-like guessing behavior based on defined game rules and exports
the results to a CSV file.

Modules
-------
- RuleHuman: Contains the rule-based guessing logic for the Wordle game.
- generator: Handles Words creation and CSV export based on the given input words.
- main: Entry point for running Words generation using the `generate_dataset()` function.

Usage
-----
Example:
--------
from DatasetPrepare.generator import generate_dataset

generate_dataset(language="tr",file_path="../Words/words_tr.txt", output_csv="turkishgamelog1000000.csv"

Parameters
----------
language : str
    "tr" for Turkish alphabet, "en" for English alphabet.
file_path : str
    Path to the file containing valid words for the selected language.
output_csv : str
    Path and name to write result of simulations

Notes
-----
- The randomness is not fixed (no seed) to ensure natural variation.
- Output CSV contains word, feedback, and guess information for each simulated game.
"""

from generator import generate_dataset

def main():
    generate_dataset(language="tr",
                     file_path="../Words/words_tr.txt",
                     output_csv="turkishgamelog1000000.csv")

if __name__ == "__main__":
    main()