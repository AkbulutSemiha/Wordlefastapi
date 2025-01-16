from Rule_Human import HumanBehaviour
import pandas as pd
import random
if __name__ == "__main__":
    file_path = "dataset/default_words.txt"
    dataset_list = []
    seen_words = set()  # Daha önce seçilen kelimeleri saklamak için set
    iteration_count = 0
    human_solver = HumanBehaviour(file_path)

    while True:
        target_word = random.choice(human_solver.words)
        first_guess = random.choice(human_solver.words)
        # Eğer hem target_word hem de first_guess daha önce seçilmişse yeni kelime seç
        if (target_word, first_guess) in seen_words:
            continue
        seen_words.add((target_word, first_guess))

        if iteration_count <= 1000000:
            dataset_list.extend(human_solver.play_wordle_like_human(target_word,first_guess, iteration_count))
            iteration_count += 1
            # Her 1000 iterasyonda dataset_list'i CSV'ye yaz
            if iteration_count % 1000000 == 0:
                df = pd.DataFrame(dataset_list,
                                  columns=['gameid', 'attempt_index', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5', 'l1', 'l2',
                                           'l3',
                                           'l4', 'l5', 'g1', 'g2', 'g3', 'g4', 'g5', 't1', 't2', 't3',
                                           't4', 't5'])
                df.to_csv("word_lists/gamelog1000000.csv", index=False, mode='a', encoding='utf-8')
                print(f"{iteration_count} iterasyonda sonuçlar CSV'ye yazıldı.")
                dataset_list = []