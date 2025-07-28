import bs4
import requests
import csv
import pandas as pd
from config import *
def extract_links(url):
    response = requests.get(url)
    response.raise_for_status()
    html_content = response.text
    soup = bs4.BeautifulSoup(html_content, "html.parser")
    links = []
    for listing in soup.find_all("div", class_="listing clearfix impression-candidate"):
        headline_sec = listing.find("div", class_="headlineSec")
        if headline_sec:
            headline = headline_sec.find("h2", class_="headline")
            if headline:
                a_tag = headline.find("a", href=True)
                if a_tag:
                    text = a_tag.get_text(strip=True)
                    href = a_tag["href"]
                    if not any(link["href"] == href for link in links):
                        links.append({"title": text, "href": href, "article_text": "", "date": ""})
    for link in links:
        try:
            article_response = requests.get(link["href"])
            article_response.raise_for_status()
            article_html = article_response.text
            article_soup = bs4.BeautifulSoup(article_html, "html.parser")
            content_sec = article_soup.find("div", class_="contentSec")
            mainArea = content_sec.find("div", class_="mainArea") if content_sec else None
            if mainArea:
                for div in mainArea.find_all("div"):
                    div.decompose()
            if mainArea:
                paragraphs = content_sec.find_all("p")
                for i, p in enumerate(paragraphs, 1):
                    para = p.get_text(strip=True)
                    if para:
                        link["article_text"] += f" {para}"
            else:
                link["article_text"] = ""
        except Exception:
            link["article_text"] = ""
    for link in links:
        try:
            article_response = requests.get(link["href"])
            article_response.raise_for_status()
            article_html = article_response.text
            article_soup = bs4.BeautifulSoup(article_html, "html.parser")
            updated_span = article_soup.find(attrs={"data-updatedtime": True})
            if updated_span:
                link["date"] = updated_span["data-updatedtime"]
            else:
                link["date"] = ""
        except Exception:
            link["date"] = ""
    links = [link for link in links if all(link[field] for field in ["title", "href", "article_text", "date"])]
    return links
def save_to_csv(webpage_link, filename, limit):
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "href", "article_text", "date"])
        writer.writeheader()
    for i in range(1, limit + 1):
        url = f"{webpage_link}/page-{i}"
        try:
            links = extract_links(url)
            for link in links:
                with open(filename, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["title", "href", "article_text", "date"])
                    writer.writerow(link)
            print(f"Iteration {i}/{limit} completed")
        except requests.RequestException as e:
            print(f"An error occurred while processing {url}: {e}")
def news_extractor_ht():
    print("Extracting links from Hindustan Times Bengal section...")
    save_to_csv(bengal_url_HT, "extracted_links_bengal_ht.csv", BENGAL_LIMIT_HT)
    print("Extracting links from Hindustan Times Sports section...")
    save_to_csv(sports_url_HT, "extracted_links_sports_ht.csv", SPORTS_LIMIT_HT)
    print("Extracting links from Hindustan Times Tech section...")
    save_to_csv(tech_url_HT, "extracted_links_tech_ht.csv", TECH_LIMIT_HT)
    print("Extracting links from Hindustan Times World section...")
    save_to_csv(world_url_HT, "extracted_links_world_ht.csv", WORLD_LIMIT_HT)
    print("Extracting links from Hindustan Times Entertainment section...")
    save_to_csv(entertainment_url_HT, "extracted_links_entertainment_ht.csv", ENT_LIMIT_HT)
    csv_files = [
        ("extracted_links_bengal_ht.csv", "bengal"),
        ("extracted_links_sports_ht.csv", "sports"),
        ("extracted_links_tech_ht.csv", "technology"),
        ("extracted_links_world_ht.csv", "world"),
        ("extracted_links_entertainment_ht.csv", "entertainment")
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
        combined_df.to_csv("extracted_links_ht.csv", index=False, encoding="utf-8")
        print("All links combined and saved to extracted_links.csv")
    else:
        print("No links found to combine.")

