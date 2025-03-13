import os
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import plotly.io as pio

available_ds = ["mooncake"]
keys = ["timestamp"]

df = pd.read_csv(".cache/traces/mooncake.csv")


def aggregate_df(df):
    results = []
    timestamps = df["timestamp"].unique()
    # for every timestamp, aggregate how many input tokens and how many output tokens
    for ts in tqdm(timestamps):
        sub_df = df[df["timestamp"] == ts]
        input_tokens = 0
        output_tokens = 0
        for row in sub_df.iterrows():
            row = row[1]
            conv_length_data = eval(row["conv_length"])
            input_seq_length = conv_length_data[0]
            output_seq_length = conv_length_data[1]
            input_tokens += input_seq_length
            output_tokens += output_seq_length
        results.append({"timestamp": ts, "tokens": input_tokens, "type": "input"})
        results.append(
            {"timestamp": ts, "tokens": input_tokens + output_tokens, "type": "Total"}
        )
    return pd.DataFrame(results)


agg_df = aggregate_df(df)
fig = px.area(agg_df, x="timestamp", y="tokens", color="type")
# set x-axis title
fig.update_xaxes(title_text="Timestamp (ms)")
fig.update_yaxes(title_text="Tokens")
# set title
fig.update_layout(title_text="Mooncake Traffic")
# save as pdf
pio.full_figure_for_development(fig, warn=False)
fig.write_image(".cache/plots/mooncake_traffic.pdf")
