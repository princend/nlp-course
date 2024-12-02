from transformers import pipeline


def sentiment_analysis(
    text="I've been waiting for a HuggingFace course my whole life.",
):
    classifier = pipeline("sentiment-analysis")
    a = classifier(text)
    print(a)


def zero_shot_classification(
    text="This is a course about the Transformers library",
    canditate_labels=["education", "politics", "business"],
):
    zeroshot = pipeline("zero-shot-classification")
    zeroshotanswer = zeroshot(
        text,
        candidate_labels=canditate_labels,
    )
    print(zeroshotanswer)
    
    
zero_shot_classification()
