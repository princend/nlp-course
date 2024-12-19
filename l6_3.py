from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
print(type(encoding))
print(tokenizer.is_fast)
print(encoding.tokens())

start, end = encoding.word_to_chars(3)
print(example[start:end])

# print(tokenizer.decode(encoding.ids))