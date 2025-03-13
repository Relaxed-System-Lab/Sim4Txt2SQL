import pandas as pd
df = pd.read_csv('notebook/all_quality.csv')
interested_model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
df = df[df['model'] == interested_model]
# pivot table, index is metrics
df = df.pivot(index='params', columns='metric', values='quality')
# reset index
df = df.reset_index()
# sort by params
df = df.sort_values('params')
# save to csv
df.to_csv('notebook/sub_quality.csv', index=False)