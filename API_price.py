from datasets import load_dataset
from itertools import chain
import math

ENG_TO_HIN_EXP_FACTOR = 1.1  # Considering a 10% expansion in number of characters after translation from English to Hindi

ds = load_dataset("CharanSaiVaddi/Hindi_FakeNews")
eng_ds = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")

print("===== Hindi to Bangla Translation =====")
org_char_len = 0
for item in ds['train']:
    text, label = item['text'], item['label']
    org_char_len += len(text)
print(f"Total characters in dataset: {org_char_len}")

print("===== English to Hindi to Bangla Translation =====")
char_len = 0
combined_eng_ds = list(chain(eng_ds['train'], eng_ds['test'], eng_ds['validation']))
for item in combined_eng_ds:
    text, label = item['text'], item['label']
    char_len += len(text)
print(f"Total characters in dataset: {char_len}")
hin_char_len = math.ceil(char_len * ENG_TO_HIN_EXP_FACTOR)
print(f"Total characters in Hindi dataset (assuming a character expansion by {ENG_TO_HIN_EXP_FACTOR}): {hin_char_len}")

print("===== Total =====")
total_chars = org_char_len + char_len + hin_char_len
print(f"Total characters across all datasets: {total_chars}")
# print(f"Total price (lower estimate) required for translation: {total_chars * PER_TOKEN_PRICE} INR")
