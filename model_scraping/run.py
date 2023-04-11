from huggingface_hub import HfApi, ModelFilter, ModelCard
from huggingface_hub.hf_api import ModelInfo
import pandas as pd

api = HfApi()


def fetch_models():
    return list(
        iter(
            api.list_models(
                # filter=ModelFilter(library="transformers"),
                sort="downloads",
                direction=-1,
                limit=30,
                fetch_config=True,
                cardData=True,
            )
        )
    )


def model_to_dict(model: ModelInfo):
    return {
        "modelId": model.modelId,
        "sha": model.sha,
        "lastModified": model.lastModified,
        "tags": model.tags,
        "pipeline_tag": model.pipeline_tag,
        "siblings": model.siblings,
        "private": model.private,
        "author": model.author,
        "likes": model.likes,
        "downloads": model.downloads,
        "config": model.config,
    }


def write_model_card(model_id: str, card_text):
    text_file = open(f"cards/{model_id.replace('/', '___')}.md", "w")
    n = text_file.write(card_text.strip())
    text_file.close()


models = fetch_models()
model_cards = [ModelCard.load(m.modelId) for m in models]
[write_model_card(m.modelId, c.text) for m, c in zip(models, model_cards)]

# model_list = [model_to_dict(m) for m in models]
# df = pd.DataFrame(model_list)
#
# df.to_json("transformers_dump.jsonl", orient="records", lines=True)
#
# df["model_type"] = df.config.apply(lambda x: x and x.get("model_type", None))
# df.model_type = df.model_type.apply(lambda x: None if x == {} else x)
# df = df.dropna(subset=["model_type"])
#
# df.model_type.value_counts()
#
# df.groupby(["model_type"]).downloads.sum().sort_values(ascending=False)
