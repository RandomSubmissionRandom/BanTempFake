import re
import unicodedata
key = "শুভেন্দু"
bengali_diacritics = ["ে", "া", "ি", "ো", "ু", "ৃ", "ৄ", "ৗ", "ৢ", "ৣ", "ৈ", "ূ", "ৌ"]
l = list(key)
output = key
for i in range(len(l) - 1, -1, -1):
    char = l[i]
    if char in bengali_diacritics:
        output = output[:i] + output[i+1:]
    elif unicodedata.category(char) == 'Lo':
        break
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(output)