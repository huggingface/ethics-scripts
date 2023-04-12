from datasets import Dataset, load_dataset

ds = load_dataset("laion/laion2B-en", split="train", streaming=True)
ds = Dataset.from_list(list(ds.take(100000)))
ds.save_to_disk("laion2B-en-small")
