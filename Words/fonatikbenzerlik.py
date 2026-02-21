# Türkçe sesli harfler
VOWELS = set("AEIİOÖUÜ")

def cv_pattern(word):
    """
    Kelimeyi C/V fonotaktik şablonuna çevirir.
    Örn: KAZAN -> CVCVC
    """
    l=""
    for ch in word.upper():
        if ch in VOWELS:
            l += "V"
        else:
            l += "C"
    return l

# --- Dosyayı oku ---
file_path = "words_tr.txt"  # kendi dosya adını buraya yaz

with open(file_path, "r", encoding="utf-8") as f:
    words = [line.strip() for line in f if len(line.strip()) == 5]

# --- Hedef kelime ---
target_word = "BAĞLI"
target_pattern = cv_pattern(target_word)

# --- Aynı fonotaktik yapıya sahip kelimeler ---
same_pattern_words = [
    w for w in words if cv_pattern(w) == target_pattern
]

print(f"Hedef kelime: {target_word}")
print(f"Fonotaktik yapı: {target_pattern}")
print(f"Aynı yapıya sahip kelime sayısı: {len(same_pattern_words)}")

# İstersen ilk 20 tanesini gör
print("Örnekler:", same_pattern_words[:20])