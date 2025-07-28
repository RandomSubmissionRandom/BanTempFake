import json
import random
import re
def extract_placeholders(template):
    pattern = r"<([A-Z\-0-9_]+)\|([^|]*)\|([0-9]+)>"
    matches = re.findall(pattern, template)
    return [{"label": m[0], "cutoff": m[1], "reference_number": int(m[2])} for m in matches]
def generate_faked_text(template, entity_map, entities, random_seed=16032005):
    random.seed(random_seed)
    current_keys = set(entity_map.keys())
    tag_to_entities = {}
    for key in entities.keys():
        if key not in current_keys:
            for tag in entities[key]:
                tag_to_entities.setdefault(tag, []).append(key)
    new_entity_map = {}
    used_entities = set()
    for old_key, (idx, tag) in entity_map.items():
        candidates = [e for e in tag_to_entities.get(tag, []) if e not in used_entities]
        candidates = [e for e in candidates if not any('a' <= ch.lower() <= 'z' for ch in e)]
        if candidates:
            new_key = random.choice(candidates)
            used_entities.add(new_key)
            new_entity_map[new_key] = [idx, tag]
        else:
            new_entity_map[old_key] = [idx, tag]
    placeholders = extract_placeholders(template)
    faked_text = template
    for ph in placeholders:
        idx = ph["reference_number"]
        tag = ph["label"]
        cutoff = ph["cutoff"]
        key = next((k for k, v in new_entity_map.items() if v[0] == idx), None)
        if key is not None:
            if cutoff == '\u25CC':
                replacement = key
            else:
                bengali_diacritics = ["ে", "া", "ি", "ো", "ু", "ৃ", "ৄ", "ৗ", "ৈ", "ূ", "ৌ"]
                bengali_vowels = ["অ", "আ", "ই", "ঈ", "উ", "ঊ", "এ", "ঐ", "ও", "ঔ"]
                if key and key[-1] in bengali_vowels and cutoff and cutoff[0] in bengali_diacritics:
                    remaining_cutoff = cutoff[1:] if len(cutoff) > 1 else '\u25CC'
                    replacement = key + remaining_cutoff
                else:
                    replacement = key + cutoff
            pattern = r"<{}[|]{}[|]{}>".format(re.escape(tag), re.escape(cutoff), idx)
            faked_text = re.sub(pattern, replacement, faked_text, count=1)
    return faked_text, new_entity_map
if __name__ == "__main__":
    original = """রাজ্যে ১০০ দিনের কাজের প্রকল্প ফের চালু নিয়ে কলকাতা হাইকোর্টের নির্দেশে সন্তোষ প্রকাশ করলেন রাজ্যের বিরোধী দলনেতা শুভেন্দু অধিকারী। শুক্রবার বিধানসভায় সাংবাদিকদের মুখোমুখি হয়ে তিনি দাবি, করেন বুধবারের রায়ে কেন্দ্রীয় সরকারকে ১০০ দিনের প্রকল্প পরিচালনার সমস্ত ক্ষমতা দিয়েছে আদালত। একই সঙ্গে তিনি জানান, ১০০ দিনের কাজের প্রকল্পের সমান্তরাল ভাবে চোর ধরো চোর ভরো প্রকল্প চলবে রাজ্যে। দুর্নীতি হতে দেবে না বিজেপি। এদিন শুভেন্দুবাবু বলেন, ‘আমরা প্রকল্পের পক্ষে আর চুরির বিপক্ষে। চোর ধরো জেল ধরো প্রকল্প চলবে। চুরি করতে দেওয়া যাবে না। কোর্টের নির্দেশে তো পরিষ্কার কেন্দ্রীয় সরকারের হাতে পুরো ক্ষমতা দিয়ে দিয়েছে। ভুয়ো ও প্রকৃত উপভোক্তাকে আলাদা করতে বলেছে। প্রকৃত উপভোক্তাদের কাজ দেওয়ার কথা বলেছে। আমরা তো প্রকৃত উপভোক্তাদের পক্ষে। আদালত বলেছে জব কার্ড হোল্ডারের কাছে কেন্দ্র সরাসরি টাকা পাঠাতে পারবে। রাজ্যের সহমতি ছাড়াই সরাসরি নজরদারি করতে পারবে কেন্দ্রীয় সরকার। আমরা তো প্রকল্পের পক্ষে। চুরির বিরুদ্ধে। আর হাইকোর্টের অর্ডার হয়েছে চুরি আটকাও, টাকা নেও, কাজ করো।’ বুধবার এক রায়ে কলকাতা হাইকোর্টের প্রধান বিচারপতি বিচারপতি টিএস শিবজ্ঞানমের ডি্ভিশন বেঞ্চ নির্দেশ দেয়, রাজ্যে বন্ধ থাকা ১০০ দিনের কাজের প্রকল্প ফের চালু করতে হবে কেন্দ্রকে। দুর্নীতির অভিযোগ উঠলেও এভাবে কোনও প্রকল্প একটি রাজ্যে অনির্দিষ্টকালের জন্য বন্ধ রাখা যায় না বলে মন্তব্য করেন বিচারপতিরা। তবে আদালত এও জানায় যে দুর্নীতি রুখতে রাজ্যের ওপর যে কোনও শর্ত চাপাতে পারবে কেন্দ্র। এমনকী জব কার্ড হোল্ডারদের টাকা সরাসরি পৌঁছে দিতে পারবে তাদের ব্যাঙ্ক অ্যাকাউন্টে। একই সঙ্গে রাজ্যে ১০০ দিনের কাজে দুর্নীতির সমস্ত তদন্ত জারি থাকবে বলে জানিয়েছে আদালত।"""
    template = """রাজ্যে <NUM_3|◌|0> দিনের কাজের প্রকল্প ফের চালু নিয়ে <B-ORG|◌|1> <I-ORG|ের|2> নির্দেশে সন্তোষ প্রকাশ করলেন রাজ্যের বিরোধী দলনেতা <B-PER|◌|3> <I-PER|◌|4> । শুক্রবার <B-ORG|য়|5> সাংবাদিকদের মুখোমুখি হয়ে তিনি দাবি, করেন বুধবারের রায়ে কেন্দ্রীয় সরকারকে <NUM_3|◌|0> দিনের প্রকল্প পরিচালনার সমস্ত ক্ষমতা দিয়েছে আদালত। একই সঙ্গে তিনি জানান, <NUM_3|◌|0> দিনের কাজের প্রকল্পের সমান্তরাল ভাবে চোর ধরো চোর ভরো প্রকল্প চলবে রাজ্যে। দুর্নীতি হতে দেবে না <B-ORG|◌|6> । এদিন <B-PER|◌|7> বলেন, ‘আমরা প্রকল্পের পক্ষে আর চুরির বিপক্ষে। চোর ধরো জেল ধরো প্রকল্প চলবে। চুরি করতে দেওয়া যাবে না। কোর্টের নির্দেশে তো পরিষ্কার কেন্দ্রীয় সরকারের হাতে পুরো ক্ষমতা দিয়ে দিয়েছে। ভুয়ো ও প্রকৃত উপভোক্তাকে আলাদা করতে বলেছে। প্রকৃত উপভোক্তাদের কাজ দেওয়ার কথা বলেছে। আমরা তো প্রকৃত উপভোক্তাদের পক্ষে। আদালত বলেছে জব কার্ড হোল্ডারের কাছে <B-ORG|র|8> সরাসরি টাকা পাঠাতে পারবে। রাজ্যের সহমতি ছাড়াই সরাসরি নজরদারি করতে পারবে কেন্দ্রীয় সরকার। আমরা তো প্রকল্পের পক্ষে। চুরির বিরুদ্ধে। আর <B-ORG|ের|2> অর্ডার হয়েছে চুরি আটকাও, টাকা নেও, কাজ করো। ’ বুধবার এক রায়ে <B-ORG|◌|1> <I-ORG|ের|2> প্রধান বিচারপতি বিচারপতি <B-PER|◌|9> <I-PER|ের|10> ডি্ভিশন বেঞ্চ নির্দেশ দেয়, রাজ্যে বন্ধ থাকা <NUM_3|◌|0> দিনের কাজের প্রকল্প ফের চালু করতে হবে <B-ORG|কে|11> । দুর্নীতির অভিযোগ উঠলেও এভাবে কোনও প্রকল্প একটি রাজ্যে অনির্দিষ্টকালের জন্য বন্ধ রাখা যায় না বলে মন্তব্য করেন বিচারপতিরা। তবে আদালত এও জানায় যে দুর্নীতি রুখতে রাজ্যের ওপর যে কোনও শর্ত চাপাতে পারবে <B-ORG|র|8> । এমনকী জব কার্ড হোল্ডারদের টাকা সরাসরি পৌঁছে দিতে পারবে তাদের ব্যাঙ্ক অ্যাকাউন্টে। একই সঙ্গে রাজ্যে <NUM_3|◌|0> দিনের কাজে দুর্নীতির সমস্ত তদন্ত জারি থাকবে বলে জানিয়েছে আদালত।"""
    entity_map = {"১০০": [0, "NUM_3"], "কলকাতা": [1, "B-ORG"], "হাইকোর্ট": [2, "I-ORG"], "শুভেন্দু": [3, "B-PER"], "অধিকারী": [4, "I-PER"], "বিধানসভা": [5, "B-ORG"], "বিজেপি": [6, "B-ORG"], "শুভেন্দুবাবু": [7, "B-PER"], "কেন্দ্": [8, "B-ORG"], "টিএস": [9, "B-PER"], "শিবজ্ঞানম": [10, "I-PER"], "কেন্দ্র": [11, "B-ORG"]}
    with open('entities_and_maps.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    entities = data.get('entity_to_tags', [])
    faked_text, new_entity_map = generate_faked_text(template, entity_map, entities)
    with open('final.txt', 'w', encoding='utf-8') as out_f:
        out_f.write("Original Text:\n")
        out_f.write(original)
        out_f.write("\n\nTemplate:\n")
        out_f.write(template)
        out_f.write("\n\nFaked Text:\n")
        out_f.write(faked_text)