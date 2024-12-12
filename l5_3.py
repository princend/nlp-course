from datasets import load_from_disk,load_dataset

drug_dataset_reloaded = load_from_disk("drug-reviews")
print(drug_dataset_reloaded)


data_files = {
    "train": "drug-reviews-train.jsonl",
    "validation": "drug-reviews-validation.jsonl",
    "test": "drug-reviews-test.jsonl",
}
drug_dataset_reloaded2 = load_dataset("json", data_files=data_files)
print(drug_dataset_reloaded2)