from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import TrainingArguments
from transformers import Trainer
from huggingface_hub import create_repo

# camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
# results = camembert_fill_mask("Le camembert est <mask> :)")
# print(results)


# tokenizer = AutoTokenizer.from_pretrained("camembert-base")
# model = AutoModelForMaskedLM.from_pretrained("camembert-base")

checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.push_to_hub("dummy-model")
tokenizer.push_to_hub("dummy-model")

create_repo("dummy-model")

# training_args = TrainingArguments(
#     "bert-finetuned-mrpc", save_strategy="epoch", push_to_hub=True
# )

# trainer = Trainer(
#     model,
#     training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
# )

# trainer.train()

