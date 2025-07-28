import csv
import warnings
from template_extractor import *
from fake_generator import *
from config import *
from news_extractor_ht import *
from news_extractor_gs import *
# from news_extractor_abp import *
from csv_combiner import *
from rewriter import *
import json
import pandas as pd
import sys
warnings.filterwarnings("ignore")
csv.field_size_limit(sys.maxsize)
entity_map_list = []
indexed_entity_map = {}
news_extractor_ht()
news_extractor_gs()
# news_extractor_abp()
combine_csv_files(["extracted_links_ht.csv", "extracted_links_gs.csv"], input_file)
with open(input_file, newline='', encoding='utf-8') as infile, open(output_file, 'a', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['template', 'entity_map']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in reader:
        start_row = 1
        current_row = reader.line_num
        if current_row < start_row:
            continue
        article_text = row.get('article_text', '')
        template, entity_map = obtain_template(article_text)
        row['template'] = template
        row['entity_map'] = json.dumps(entity_map, ensure_ascii=False)
        entity_map_list.append(entity_map)
        writer.writerow(row)
entity_to_tags = combine_entity_map(output_file)
fieldnames = ["title","href","article_text","date","label","template","entity_map"]
with open(full_file, 'w', newline='', encoding='utf-8') as datasetfile:
    dataset_fieldnames = fieldnames
    if 'fake_text' not in dataset_fieldnames:
        dataset_fieldnames += ['fake_text']
    if 'fake_entity_map' not in dataset_fieldnames:
        dataset_fieldnames += ['fake_entity_map']
    dataset_writer = csv.DictWriter(datasetfile, fieldnames=dataset_fieldnames)
    dataset_writer.writeheader()
    with open(output_file, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            template = row.get('template', '')
            entity_map = json.loads(row.get('entity_map', '{}'))
            article_text = row.get('article_text', '')
            fake_text, fake_entity_map = generate_faked_text(template, entity_map, entity_to_tags)
            row['fake_text'] = fake_text
            row['fake_entity_map'] = json.dumps(fake_entity_map, ensure_ascii=False)
            dataset_writer.writerow(row)
df = pd.read_csv(full_file)
true_df = pd.DataFrame({
    'text': df['article_text'],
    'label': True
})
fake_df = pd.DataFrame({
    'text': df['fake_text'],
    'label': False
})
combined_df = pd.concat([true_df, fake_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
combined_df.to_csv(dataset_file, index=False)
combined_df = pd.read_csv(dataset_file)
# with open(rewritten_dataset_file, 'w', newline='', encoding='utf-8') as rewritten_file:
#     fieldnames = list(combined_df.columns) + ['rewritten_text']
#     writer = csv.DictWriter(rewritten_file, fieldnames=fieldnames)
#     writer.writeheader()
#     for _, row in combined_df.iterrows():
#         row_dict = row.to_dict()
#         row_dict['rewritten_text'], _ = rewrite_text(row_dict['text'])
#         writer.writerow(row_dict)