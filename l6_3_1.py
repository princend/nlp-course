from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
todevice = "cuda" if torch.cuda.is_available() else "cpu"


token_classifier = pipeline("token-classification", device=todevice)
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")

# token_classifier = pipeline("token-classification", aggregation_strategy="simple")
# token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
model.to(todevice)
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)

# print(inputs["input_ids"].shape)
# print(outputs.logits.shape)


probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
# print(predictions)


results = []
tokens = inputs.tokens()

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        results.append(
            {"entity": label, "score": probabilities[idx][pred], "word": tokens[idx]}
        )

# print(results)

inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
inputs_with_offsets["offset_mapping"]

example[12:14]


results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        start, end = offsets[idx]
        results.append(
            {
                "entity": label,
                "score": probabilities[idx][pred],
                "word": tokens[idx],
                "start": start,
                "end": end,
            }
        )

print(results)