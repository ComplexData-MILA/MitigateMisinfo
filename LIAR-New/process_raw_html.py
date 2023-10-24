'''
A script to process the raw HTML from PolitiFact into an unannotated version of LIAR-New.
'''

import pickle
import re
from bs4 import BeautifulSoup
import pandas as pd

with open('LIAR_new_rawscrape.pkl', 'rb') as f:
  data = pickle.load(f)

def remove_after_number(s):
    result = re.match(r".*?\b(\d{4})\b", s)
    if result:
        return result.group()
    else:
        return s

processed_data_list = []

for key, example in data.items():
  try:
    soup = BeautifulSoup(example.content, 'html.parser')

    statement = soup.find('div', {'class': 'm-statement__quote'}).text.strip()

    img_tags = soup.find_all('img')
    label = img_tags[4].get('alt')

    dates = soup.findAll("div", {"class": "m-statement__desc"})
    date = dates[0].text.split('on ')[1].split(' in')[0]
    date = remove_after_number(date)

    processed_data_list.append([key, statement, label, date])
  except:
    continue
  
df = pd.DataFrame(processed_data_list, columns=['example_id', 'statement', 'label', 'date'])

print(df.label.value_counts())
'''
false          1067
pants-fire      359
barely-true     237
half-true       147
mostly-true      99
true             48
full-flop         7
half-flip         5
'''

def remove_outer_quotes(input):
  if input[0] == "\"" and input[-1] == "\"":
    return input[1:-1]
  if input[0] == "“" and input[-1] == "”":
    return input[1:-1]
  if input[0] == "“" and input[-1] == "\"":
    return input[1:-1]
  else:
    return input

df_filtered = df[(df.label != 'full-flop') & (df.label != 'half-flip')]
#df_filtered['statement'] = df_filtered.statement.apply(remove_outer_quotes)

df_filtered.to_csv('LIAR-New_unannotated.tsv', sep='\t', index=False)




# ARTICLE TEXT
# (for Web Oracle)

with open('/network/scratch/k/kellin.pelrine/LIAR_new_rawscrape.pkl', 'rb') as f:
    scraped_data = pickle.load(f)

article_dict = {}
for key, test_example in scraped_data.items():
    try:
        html = test_example.text

        soup = BeautifulSoup(html, 'html.parser')

        article = soup.find('article', {'class': 'm-textblock'})

        paragraphs = article.find_all('p')

        text = ' '.join([p.get_text() for p in paragraphs]).replace('\xa0', ' ')

        article_dict[key] = text
    except:
        continue
  
df = pd.DataFrame.from_dict(article_dict, orient='index', columns=['article_text'])
df = df.reset_index().rename(columns={'index': 'example_id'})

print(len(df))

df.to_json('LIAR_new_scrapedarticles.jsonl', lines=True, orient='records')