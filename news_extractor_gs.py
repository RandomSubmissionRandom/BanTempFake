import bs4
import requests
import csv
import pandas as pd
from config import *
def extract_links(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.content, 'html.parser')
    links = []
    grid_box = soup.find(class_='grid-box')
    if grid_box:
        rows = grid_box.find_all(class_='row')
        for row in rows:
            post_titles = row.find_all(class_='post-title')
            for post_title in post_titles:
                h2_tag = post_title.find('h2')
                if h2_tag:
                    a_tag = h2_tag.find('a')
                    if a_tag:
                        title = a_tag.get_text(strip=True)
                        href = a_tag.get('href')
                        date = ""
                        post_tags = post_title.find_next(class_='post-tags')
                        if post_tags:
                            first_li = post_tags.find('li')
                            if first_li:
                                date = first_li.get_text(strip=True)
                        links.append({
                            'title': title,
                            'href': href,
                            'article_text': "",
                            'date': date
                        })
    for link in links:
        try:
            article_response = requests.get(link['href'])
            article_response.raise_for_status()
            article_soup = bs4.BeautifulSoup(article_response.content, 'html.parser')
            single_post_content = article_soup.find(class_='single-post-box')
            if single_post_content:
                post_content = single_post_content.find(class_='post-content')
                if post_content:
                    p_tags = post_content.find_all('p')
                    article_text = ""
                    for p in p_tags:
                        for br in p.find_all('br'):
                            br.decompose()
                        article_text += p.get_text(strip=True) + " "
                    link['article_text'] = article_text.strip()
        except requests.RequestException:
            pass
    return links
def save_to_csv(webpage_link, filename, limit):
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "href", "article_text", "date"])
        writer.writeheader()
    for i in range(1, limit + 1):
        url = f"{webpage_link}?page={i}"
        try:
            links = extract_links(url)
            for link in links:
                with open(filename, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["title", "href", "article_text", "date"])
                    writer.writerow(link)
            print(f"Iteration {i}/{limit} completed")
        except requests.RequestException as e:
            print(f"An error occurred while processing {url}: {e}")
def news_extractor_gs():
    print("Extracting links from Ganashakti National section...")
    save_to_csv(national_url_GS, "extracted_links_national_gs.csv", NATIONAL_LIMIT_GS)
    print("Extracting links from Ganashakti Bengal section...")
    save_to_csv(bengal_url_GS, "extracted_links_bengal_gs.csv", BENGAL_LIMIT_GS)
    print("Extracting links from Ganashakti Sports section...")
    save_to_csv(sports_url_GS, "extracted_links_sports_gs.csv", SPORTS_LIMIT_GS)
    print("Extracting links from Ganashakti World section...")
    save_to_csv(world_url_GS, "extracted_links_world_gs.csv", WORLD_LIMIT_GS)
    csv_files = [
        ("extracted_links_national_gs.csv", "national"),
        ("extracted_links_sports_gs.csv", "sports"),
        ("extracted_links_world_gs.csv", "world"),
        ("extracted_links_bengal_gs.csv", "bengal")
    ]
    all_links = []
    for file, label in csv_files:
        try:
            df = pd.read_csv(file)
            df["label"] = label
            all_links.append(df)
        except FileNotFoundError:
            continue
    if all_links:
        combined_df = pd.concat(all_links, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["href"])
        combined_df.to_csv("extracted_links_gs.csv", index=False, encoding="utf-8")
        print("All links combined and saved to extracted_links.csv")
    else:
        print("No links found to combine.")
if __name__ == "__main__":
    news_extractor_gs()
