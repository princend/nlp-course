import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets   import load_dataset

def processing_data():

    checkpoint = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequences = [
        "I've been waiting for a HuggingFace course my whole life.",
        "This course is amazing!",
    ]

    batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

    # This is new
    batch["labels"] = torch.tensor([1, 1])

    optimizer = AdamW(model.parameters())

    outputs = model(**batch)
    loss = outputs.loss

    loss.backward()

    optimizer.step()
    print(outputs)
    print(outputs.logits)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)

    print(outputs.logits)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)

def load_datasets():
    raw_datas=load_dataset('glue','mrpc')
    print(raw_datas)
    
    raw_train_dataset = raw_datas["train"]
    data = raw_train_dataset[0]
    print(data)

def tokenize_data():
    checkpoint = 'bert-base-uncased'
    tokenizer=AutoTokenizer.from_pretrained(checkpoint)
    raw_datasets=load_dataset('glue','mrpc')
    tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
    tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

def tokenize_multi_input():
    checkpoint = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer("This is the first sentence.", "This is the second one.")
    print(inputs)
    
    converted_sentence =tokenizer.convert_ids_to_tokens(inputs["input_ids"])
    print(converted_sentence)
    
def tokenize_function(example):
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)    


def tokenize_batch():
    raw_datasets=load_dataset('glue','mrpc')
    # batched=True 表示将多个样本一起处理 會把長度不一的句子padding成一樣的長度
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    print(tokenized_datasets)

# samples = {
#     "idx": 1,
#     "sentence1": "Hello",
#     "sentence2": "World",
#     "label": "greeting"
# }

# samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

# samples = {
#     "label": "greeting"
# }

def data_collator():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    raw_datasets=load_dataset('glue','mrpc')
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    samples = tokenized_datasets["train"][:8]
    samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
    list_len = [len(x) for x in samples["input_ids"]]
    print(list_len)
    batch = data_collator(samples)
    print({k: v.shape for k, v in batch.items()})
# processing_data()
# load_datasets()
# tokenize_data()
# tokenize_multi_input()
data_collator()