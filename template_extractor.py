from stemmer import *
from ner import *
from kulemma import *
import csv
import ast
PUNCS = ['।', '!', '?', ',', ';', ':', '(', ')', '[', ']', '{', '}', '"', "'", '—', '–', '+', '-', '*', '/', '\\', '%', '=', '>', '<', '@', '#', '$', '^', '&', '_', '~', "", " "]
def preprocess(text):
    text = re.sub(r'(?<!\s)([' + re.escape(''.join(PUNCS)) + '])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'([০-৯0-9])(?=[^\s০-৯0-9])', r'\1 ', text)
    return text
def obtain_template(input_text, lemma=False):
    template = ''
    entity_map = {}
    text = preprocess(input_text)
    metadata = {}
    ner_result = get_ner(text, metadata=metadata)
    BASE = metadata.get('base', 'O')
    for i, (word, tag) in enumerate(ner_result):
        if (word not in PUNCS):
            if bool(re.match(r'^[০-৯0-9]+$', word)):
                ner_result[i] = (word, f'NUM_{len(word)}')
                tag = f'NUM_{len(word)}'
            if (tag != BASE):
                if lemma:
                    stemmed_word = PosLemmatizer().lemmatize(word)
                else:
                    stemmed_word = BanglaStemmer().stem(word)
                if (stemmed_word in PUNCS) or (not stemmed_word) or (stemmed_word == '\u25CC') or (len(stemmed_word) <= 1):
                    stemmed_word = word
                if stemmed_word not in entity_map.keys():
                    entity_map[stemmed_word] = (len(list(entity_map.keys())), tag)
                placeholder = [tag]
                cutoff = word.replace(stemmed_word, '', 1)
                if word == stemmed_word:
                    cutoff = '\u25CC'
                placeholder.append(cutoff)
                placeholder.append(str(entity_map[stemmed_word][0]))
                template += '<' + '|'.join(placeholder) + '> '
            else:
                template += word + ' '
        else:
            template += word + ' '
    template = re.sub(r'\s+([' + re.escape(''.join(PUNCS)) + '])', r'\1', template)
    template = re.sub(r'(?<! )<', r' <', template)
    template = re.sub(r'>', '> ', template)
    template = re.sub(r'\s+', ' ', template)
    return (template.strip(), entity_map)
def combine_entity_map(template_csv_file):
    entity_map_list = []
    with open(template_csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            entity_map_str = row['entity_map']
            entity_map = ast.literal_eval(entity_map_str)
            entity_map_list.append(entity_map)
    all_entities = set()
    for entity_map in entity_map_list:
        all_entities.update(entity_map.keys())
    all_entities = sorted(all_entities)
    entity_to_tags = {entity: set() for entity in all_entities}
    for entity_map in entity_map_list:
        for entity, value in entity_map.items():
            if entity in entity_to_tags:
                entity_to_tags[entity].add(value[-1])
    entity_to_tags = {k: list(v) for k, v in entity_to_tags.items()}
    return entity_to_tags