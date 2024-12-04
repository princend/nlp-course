from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch 
def classfication():
    classifier = pipeline("sentiment-analysis")
    output = classifier(
        [
            "I've been waiting for a HuggingFace course my whole life.",
            "I hate this so much!",
        ]
    )
    print(output)


def tokenizer():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
        "today is good day"
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)

def automodel_for_seq_classification():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
        "今天天氣真好"
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    print(outputs.logits.shape)
    print(outputs.logits)
    predictions=torch.nn.functional.softmax(outputs.logits,dim=-1)
    print(predictions)
    print(model.config.id2label)
    
from transformers import BertConfig, BertModel


def set_bert_config():
    config = BertConfig()
    model = BertModel(config)
    print(config)

def get_pretrained_model():
    model=BertModel.from_pretrained("bert-base-cased")
    model.save_pretrained("temp-weight")


def inference():
    sequences = ["Hello!", "Cool.", "Nice!"]
    encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
    ]
    model_inputs = torch.tensor(encoded_sequences)
    print(model_inputs)
    model=BertModel.from_pretrained("bert-base-cased")
    output = model(model_inputs)
    print(output)
# classfication()
# tokenizer()
# automodel_for_seq_classification()
# set_bert_config()
# get_pretrained_model()
inference()