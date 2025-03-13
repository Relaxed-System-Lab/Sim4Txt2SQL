import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
traces = [x for x in os.listdir(".cache/traces") if x.endswith(".csv")]


def aggregate_df(df):
    models = df["model"].unique()
    data = []
    for model in models:
        sub_df = df[df["model"] == model]
        one_turn_dialogs = sub_df[sub_df["turn"] == 1]
        conv_length_data = one_turn_dialogs["conv_length"]
        for row in conv_length_data:
            conv_length_data = eval(row)
            if len(conv_length_data) > 1:
                input_seq_length = conv_length_data[0]
                output_seq_length = conv_length_data[1]
                data.append(
                    {
                        "model": model,
                        "input_length": input_seq_length,
                        "output_length": output_seq_length,
                    }
                )
    agg_df = pd.DataFrame(data)
    return agg_df


for trace in traces:
    print(f"Plotting {trace}")
    df = pd.read_csv(f".cache/traces/{trace}")
    agg_df = aggregate_df(df)
    # randomly sample 5 models
    selected_models = agg_df["model"].unique()
    print(f"Available Models: {selected_models}")
    if selected_models.size > 5:
        selected_models = selected_models[:5]
    selected_df = agg_df[agg_df["model"].isin(selected_models)]
    # if there are more than 10000 rows, sample 10000 rows
    if selected_df.shape[0] > 10000:
        selected_df = selected_df.sample(10000)
    # set the scatter larger
    sns.scatterplot(
        data=selected_df, x="input_length", y="output_length", hue="model", s=50
    )
    # set title
    plt.title(
        f"Input Length vs Output Length for {trace.removesuffix('.csv')} trace",
        fontsize=18,
    )
    # enlarge the texts
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    # enlarge axis labels
    plt.xlabel("Input Length", fontsize=16)
    plt.ylabel("Output Length", fontsize=16)
    plt.savefig(f".cache/plots/{trace}.pdf", format="pdf", bbox_inches="tight")
    # clear
    plt.clf()
