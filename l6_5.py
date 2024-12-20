from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))

## normalize
#標準化步驟涉及一些常規清理，例如刪除不必要的空格、小寫和/或刪除重音符號。如果你熟悉Unicode normalization（例如 NFC 或 NFKC），這也是 tokenizer 可能應用的東西。
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))

tokened_str=tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
print(tokened_str)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_str= tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
t5_str =tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")