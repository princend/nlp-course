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

def tokenize():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    sequence = "Using a Transformer network is simple"
    tokens = tokenizer.tokenize(sequence)

    print(tokens)    
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(ids)

def decoding():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
    print(decoded_string)   

def batch_input():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequence = "I've been waiting for a HuggingFace course my whole life."

    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([ids])
    print("Input IDs:", input_ids)
    output=model(input_ids)   
    print("Logits:", output.logits)
    
def pad_token():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequence1_ids = [[200, 200, 200]]
    sequence2_ids = [[200, 200]]
    batched_ids = [
        [200, 200, 200],
        [200, 200, tokenizer.pad_token_id],
    ]

    print(model(torch.tensor(sequence1_ids)).logits)
    print(model(torch.tensor(sequence2_ids)).logits)
    print(model(torch.tensor(batched_ids)).logits)    

def attension_mask():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    
    batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
    ]

    attention_mask = [
        [1, 1, 1],
        [1, 1, 0],
    ]

    outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
    print(outputs.logits) 
    
def truncation():
    sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

    # Will truncate the sequences that are longer than the model max length
    # (512 for BERT or DistilBERT)
    model_inputs = tokenizer(sequences, truncation=True)

    # Will truncate the sequences that are longer than the specified max length
    model_inputs = tokenizer(sequences, max_length=8, truncation=True) 
    
def ending():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

    tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    output = model(**tokens)
    print(output)
    
# classfication()
# tokenizer()
# automodel_for_seq_classification()
# set_bert_config()
# get_pretrained_model()
# inference()
# tokenize()
# decoding()
# batch_input()
# pad_token()
# attension_mask()
ending()