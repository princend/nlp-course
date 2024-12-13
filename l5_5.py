from itertools import islice
from datasets import interleave_datasets
from datasets import load_dataset

data_files = "big_dataset\PUBMED_title_abstracts_2019_baseline.jsonl.zip"

pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)

# law_dataset_streamed = load_dataset(
#     "json",
#     data_files="https://the-eye.eu/public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst",
#     split="train",
#     streaming=True,
# )
# next(iter(law_dataset_streamed))


# combined_dataset = interleave_datasets([pubmed_dataset_streamed, law_dataset_streamed])
combined_dataset = interleave_datasets([pubmed_dataset_streamed])
a=list(islice(combined_dataset, 2))

print(a)


### 資料集搬走了
# base_url = "https://the-eye.eu/public/AI/pile/"
# data_files = {
#     "train": [base_url + "train/" + f"{idx:02d}.jsonl.zst" for idx in range(30)],
#     "validation": base_url + "val.jsonl.zst",
#     "test": base_url + "test.jsonl.zst",
# }
# pile_dataset = load_dataset("json", data_files=data_files, streaming=True)
# next(iter(pile_dataset["train"]))