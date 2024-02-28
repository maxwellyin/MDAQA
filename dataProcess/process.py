# %%
import json
import pandas as pd
from datasets import Dataset, load_dataset
# %%
# "SQuAD" "NewsQA" "NaturalQuestionsShort" "HotpotQA" "TriviaQA-web" "SearchQA" "BioASQ" DuoRC
DOMAIN = "BioASQ"
SPLIT = "test" # train test
DATA = f"../../datasets/QVE/data/{DOMAIN}.{SPLIT}.json"
OUT = f"../data/{DOMAIN}_{SPLIT}"
# %%
f = open(DATA)
# %%
data = json.load(f)
# %%
print(data.keys())
print(type(data['data']))
print(len(data['data']))
# %%
df = pd.DataFrame(data['data'])
# %%
df2 = df['paragraphs']
# %%
for i in range(len(df2)):
    df2[i] = df2[i][0]
# %%
df3 = pd.DataFrame(df2.tolist())
# %%
df4 = pd.DataFrame(sum(df3['qas'], []))
# %%
nums = [len(i) for i in df3['qas']]
# %%
contexts = []
for i in range(len(nums)):
    context = df3.loc[i, 'context']
    contexts += [context] * nums[i]
# %%
df4['context'] = contexts
# %%
for i in range(len(df4)):
    df4.at[i, 'answers'] = df4.at[i, 'answers'][0]
    df4.at[i, 'answers']['answer_start'] = [df4.at[i, 'answers']['answer_start']]
    df4.at[i, 'answers']['text'] = [df4.at[i, 'answers']['text']]
# %%
dataset = Dataset.from_pandas(df4)
# %%
dataset.save_to_disk(OUT)
# %%
