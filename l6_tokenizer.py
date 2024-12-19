from datasets import load_dataset
from transformers import AutoTokenizer

# This can take a few minutes to load, so grab a coffee or tea while you wait!
raw_datasets = load_dataset("code_search_net", "python",trust_remote_code=True)

# print(raw_datasets["train"])

# print(raw_datasets["train"][123456]["whole_func_string"])

training_corpus = (
    raw_datasets["train"][i : i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)

def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )
training_corpus = get_training_corpus()
print(training_corpus)


old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
tokenizer.save_pretrained("code-search-net-tokenizer")
tokenizer.push_to_hub("code-search-net-tokenizer")


tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
print(tokenizer)