import bs4
import requests
import csv
import pandas as pd
from config import *
def extract_links(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.content, "html.parser")
    section_container = soup.find("div", class_="sectioncontainer mt-16")
    if not section_container:
        return []
    links = []
    li_elements = section_container.find_all("li")
    for li in li_elements:
        imgntextbox = li.find("div", class_="imgntextbox")
        if not imgntextbox:
            continue
        a_tag = imgntextbox.find("a")
        if not a_tag or not a_tag.get("href"):
            continue
        article_link = a_tag.get("href")
        contentbox = imgntextbox.find("div", class_="contentbox")
        if not contentbox:
            continue
        h2_tag = contentbox.find("h2")
        title = h2_tag.get_text(strip=True) if h2_tag else ""
        span_tag = contentbox.find("span")
        if span_tag:
            temp_date = span_tag.get_text(strip=True)
            date = temp_date.replace("শেষ আপডেট: ", "") if temp_date.startswith("শেষ আপডেট: ") else temp_date
        else:
            date = ""
        links.append({
            "title": title,
            "href": article_link,
            "article_text": "",
            "date": date
        })
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
def news_extractor_abp():
    print("Extracting links from Anandabazar Patrika National section...")
    save_to_csv(national_url_ABP, "extracted_links_national_abp.csv", NATIONAL_LIMIT_ABP)
    print("Extracting links from Anandabazar Patrika Bengal section...")
    save_to_csv(bengal_url_ABP, "extracted_links_bengal_abp.csv", BENGAL_LIMIT_ABP)
    print("Extracting links from Anandabazar Patrika Sports section...")
    save_to_csv(sports_url_ABP, "extracted_links_sports_abp.csv", SPORTS_LIMIT_ABP)
    print("Extracting links from Anandabazar Patrika World section...")
    save_to_csv(world_url_ABP, "extracted_links_world_abp.csv", WORLD_LIMIT_ABP)
    print("Extracting links from Anandabazar Patrika Entertainment section...")
    save_to_csv(entertainment_url_ABP, "extracted_links_entertainment_abp.csv", ENT_LIMIT_ABP)
    csv_files = [
        ("extracted_links_national_abp.csv", "national"),
        ("extracted_links_sports_abp.csv", "sports"),
        ("extracted_links_world_abp.csv", "world"),
        ("extracted_links_bengal_abp.csv", "bengal"),
        ("extracted_links_entertainment_abp.csv", "entertainment")
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
        combined_df.to_csv("extracted_links_abp.csv", index=False, encoding="utf-8")
        print("All links combined and saved to extracted_links.csv")
    else:
        print("No links found to combine.")
if __name__ == "__main__":
    news_extractor_abp()
