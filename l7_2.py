from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from evaluate import load
import numpy as np
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from huggingface_hub import Repository, get_full_repo_name


raw_datasets = load_dataset("conll2003",trust_remote_code=True)
ner_feature = raw_datasets["train"].features["ner_tags"]
pos_feature =raw_datasets["train"].features["pos_tags"]
label_names = ner_feature.feature.names
pos_label_names =pos_feature.feature.names
words = raw_datasets["train"][0]["tokens"]
labels = raw_datasets["train"][0]["ner_tags"]
pos_labels = raw_datasets["train"][0]["pos_tags"]
line1 = ""
line2 = ""

pos_line1=""
pos_line2=""

# for word, label in zip(words, labels):
#     full_label = label_names[label]
#     max_length = max(len(word), len(full_label))
#     line1 += word + " " * (max_length - len(word) + 1)
#     line2 += full_label + " " * (max_length - len(full_label) + 1)
    
    
# for word, label in zip(words, pos_labels):
#     full_label = pos_label_names[label]
#     max_length = max(len(word), len(full_label))
#     pos_line1 += word + " " * (max_length - len(word) + 1)
#     pos_line2 += full_label + " " * (max_length - len(full_label) + 1)
    
# print(line1)
# print(line2)    

# print(pos_line1)
# print(pos_line2)


model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# print(tokenizer.is_fast)

inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
inputs.tokens()

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
# batch["labels"]

# for i in range(2):
#     print(tokenized_datasets["train"][i]["labels"])
    


# metric = load_metric("seqeval")    
metric = load("seqeval")
 
labels = raw_datasets["train"][0]["ner_tags"]
labels = [label_names[i] for i in labels]

predictions = labels.copy()
predictions[2] = "O"
metric = metric.compute(predictions=[predictions], references=[labels])
# print(metric)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
    
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}    

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)


args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()

trainer.push_to_hub(commit_message="Training complete")


train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
)

# model = AutoModelForTokenClassification.from_pretrained(
#     model_checkpoint,
#     id2label=id2label,
#     label2id=label2id,
# )


# optimizer = AdamW(model.parameters(), lr=2e-5)


# accelerator = Accelerator()
# model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
#     model, optimizer, train_dataloader, eval_dataloader
# )

# num_train_epochs = 3
# num_update_steps_per_epoch = len(train_dataloader)
# num_training_steps = num_train_epochs * num_update_steps_per_epoch
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps,
# )

# model_name = "bert-finetuned-ner-accelerate"
# repo_name = get_full_repo_name(model_name)
# output_dir = "bert-finetuned-ner-accelerate"
# repo = Repository(output_dir, clone_from=repo_name)

# def postprocess(predictions, labels):
#     predictions = predictions.detach().cpu().clone().numpy()
#     labels = labels.detach().cpu().clone().numpy()

#     # Remove ignored index (special tokens) and convert to labels
#     true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
#     true_predictions = [
#         [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     return true_labels, true_predictions