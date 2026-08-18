import argparse
import os
import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape ESCO level-4 skill concepts -> data/esco_hierarchy.csv")
    parser.add_argument('--source', default='data/ESCO/skillsHierarchy_en.csv', type=str)
    parser.add_argument('--out', default='data/esco_hierarchy.csv', type=str)
    parser.add_argument('--errors', default='data/error.csv', type=str)
    parser.add_argument('--sleep', default=5.0, type=float,
                        help='seconds to wait after each page load (default: 5)')
    return parser.parse_args()


def is_nan(d):
    return pd.isna(d)


def main():
    args = parse_args()

    source = resolve(args.source)
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"ESCO hierarchy not found: {source}\n"
            f"Download the ESCO CSV distribution first; see data/README.md."
        )
    datas = pd.read_csv(source).values.tolist()

    level0 = "skills"
    level1, level1_url = "", ""
    level2, level2_url = "", ""
    levels4 = []
    results, err_lists = [], []
    results_columns = ["level0", "level1", "level1_url", "level2", "level2_url",
                       "level3", "level3_url", "level4"]
    err_columns = ["idx", "error"]

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        for idx, data in enumerate(datas):
            try:
                if is_nan(data[4]):        # level-1 row
                    level1, level1_url = data[3], data[2]
                    continue
                elif is_nan(data[6]):      # level-2 row
                    level2, level2_url = data[5], data[4]
                    continue
                level3, level3_url = data[7], data[6]
                driver.get(level3_url)
                time.sleep(args.sleep)
                try:
                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    narrower_concepts_div = soup.find("div", id="narrower-concepts-list")
                    if narrower_concepts_div:
                        text = narrower_concepts_div.get_text(separator="\n", strip=True)
                        levels4 = text.split("\n")
                except Exception as e:
                    print("Error fetching skill title:", e)

                for level4 in levels4:
                    results.append([level0, level1, level1_url, level2, level2_url,
                                    level3, level3_url, level4])
            except Exception as e:
                err_lists.append([idx, e])
                print(e, idx)
    finally:
        driver.quit()

    out_path, err_path = resolve(args.out), resolve(args.errors)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(results, columns=results_columns).to_csv(out_path, index=False, encoding="utf-8")
    pd.DataFrame(err_lists, columns=err_columns).to_csv(err_path, index=False, encoding="utf-8")
    print(f"Wrote {out_path} ({len(results)} rows) and {err_path} ({len(err_lists)} errors)")


if __name__ == "__main__":
    main()
