import pandas as pd
from huggingface_hub import HfApi

df = pd.read_csv("spaces.csv")
# spaces = df[df.Status.isna()]

hf_api = HfApi()


def get_likes(row):
    URL = row["URL"]
    space_name = URL.split("/")[-1]
    space_author = URL.split("/")[-2]
    return hf_api.list_spaces(author=space_author, search=space_name)[0].likes


df["likes"] = df[df.Status.isna()].apply(get_likes, axis=1)

print(df)

df.to_csv("./spaces_with_likes.csv")
