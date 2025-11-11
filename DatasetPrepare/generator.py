import pandas as pd
import random
from Rule_Human import HumanBehaviour
import os

def save_csv_safely(df, path):
    # Dosya var mı kontrol et
    file_exists = os.path.isfile(path)

    # Header sadece dosya yoksa eklenecek
    df.to_csv(path, mode='a', index=False, header=not file_exists)
def generate_dataset(language="tr",file_path="../Words/words_tr.txt", output_csv="turkishgamelog1000000.csv"):
    dataset_list = []
    iteration_count = 0
    human_solver = HumanBehaviour(file_path, language)

    while True:
        target_word = random.choice(human_solver.words)
        first_guess = random.choice(human_solver.words)


        if iteration_count <= 1000000:
            dataset_list.extend(
                human_solver.play_wordle_like_human(target_word, first_guess, iteration_count)
            )
            iteration_count += 1

            if iteration_count % 1000 == 0:
                df = pd.DataFrame(
                    dataset_list,
                    columns=[
                        'gameid', 'attempt_index',
                        'pg1', 'pg2', 'pg3', 'pg4', 'pg5',
                        'l1', 'l2', 'l3', 'l4', 'l5',
                        'g1', 'g2', 'g3', 'g4', 'g5',
                        't1', 't2', 't3', 't4', 't5'
                    ]
                )
                save_csv_safely(df, output_csv)
                print(f"{iteration_count} iterasyonda sonuçlar CSV'ye yazıldı.")
                dataset_list = []
        else:
            break
