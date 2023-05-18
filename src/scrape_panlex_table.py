import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unicodedata import normalize
from sys import argv

src_lang = sys.argv[1]
out_lang = sys.argv[2]
url = sys.argv[3]
output_directory = sys.argv[4]


def scrape_panlex_tables(URL, pages=1, columns=[src_lang, out_lang]):
    table = pd.DataFrame([])
    for i in range(pages):
        scraped_tables = pd.read_html(URL+str(i),)
        if scraped_tables:
            table = pd.concat([table, scraped_tables[0]], axis=0)
    return table.rename(columns={'Unnamed: 0': columns[0], 'Unnamed: 1': columns[1]})


URL = str(url)


table = scrape_panlex_tables(URL, 300, columns=[src_lang, out_lang])
# set lowercase
table_MN[src_lang] = table_MN[src_lang].apply(
    lambda x: x.lower())  
# set lowercase
table_MN[out_lang] = table_MN[out_lang].apply(
    lambda x: x.lower())  
# remove empty and duplicates
table_MN = table.dropna().drop_duplicates()  
table_MN = table_MN[table_MN[src_lang] !=
                    table_MN[out_lang]]  # no identical pairs

# no multiwords in src
table_MN = table_MN[~table_MN[src_lang].str.contains(' ')]
# no multiwords in tgt
table_MN = table_MN[~table_MN[out_lang].str.contains(' ')]

table_MN = table_MN[~((table_MN[src_lang] == 'Tagalog') & (
    table_MN[out_lang] == 'English'))]  # change this!!!
table_MN.to_csv(f'{sys.argv[4]}\\{src_lang}_{out_lang}_dict.txt',
                sep=' ', index=False, header=False)
table_MN[[out_lang, src_lang]].to_csv(
    f'{sys.argv[4]}\\{out_lang}_{src_lang}_dict.txt', sep=' ', index=False, header=False)
