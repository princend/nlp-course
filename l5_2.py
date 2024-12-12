from datasets import load_dataset
import html
from transformers import AutoTokenizer
from datasets import Dataset

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
# print(drug_sample[:3])

for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))
    
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
# print(drug_dataset)


def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

def filter_nones(x):
    return x["condition"] is not None

# drug_dataset.filter(filter_nones)
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)
drug_dataset.map(lowercase_condition)
# print(drug_dataset["train"]["condition"][:3])


def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

drug_dataset = drug_dataset.map(compute_review_length)
# Inspect the first training example
# print(drug_dataset["train"][0])

drug_dataset["train"].sort("review_length")
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
# print(drug_dataset.num_rows)


text = "I&#039;m a transformer called BERT"
unescaped_text= html.unescape(text)

# print(unescaped_text)

new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)


# print(new_drug_dataset)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)


def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    
result = tokenize_and_split(drug_dataset["train"][0])
# print([len(inp) for inp in result["input_ids"]]) 

# tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)

# tokenized_dataset = drug_dataset.map(
#     tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
# )

# print(len(tokenized_dataset["train"]), len(drug_dataset["train"]))

def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result

tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
# print(tokenized_dataset)





drug_dataset.set_format("pandas")

# print(drug_dataset["train"][:3])


train_df = drug_dataset["train"][:]


frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "condition": "frequency"})
)

# print(frequencies.head())

freq_dataset = Dataset.from_pandas(frequencies)
print(freq_dataset)

drug_dataset.reset_format()

drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
# print(drug_dataset_clean)

### 以 Arrow 格式保存我們清洗過的數據集
# drug_dataset_clean.save_to_disk("drug-reviews")

for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")