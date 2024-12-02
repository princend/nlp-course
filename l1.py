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


class TextGeneratorBase:
    def __init__(self, model, max_length=30, num_return_sequences=2):
        self.generator = pipeline("text-generation", model=model)
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences

    def generate_text(self, text):
        result = self.generator(
            text,
            max_length=self.max_length,
            num_return_sequences=self.num_return_sequences,
        )
        print(result)

class EnglishTextGenerator(TextGeneratorBase):
    def __init__(self, model="distilgpt2"):
        super().__init__(model=model)

class ChineseTextGenerator(TextGeneratorBase):
    def __init__(self, model="uer/gpt2-chinese-cluecorpussmall"):
        super().__init__(model=model)



# sentiment_analysis()
# zero_shot_classification()
# EnglishTextGenerator().generate_text("I've been waiting for a HuggingFace course my whole life.")
ChineseTextGenerator().generate_text("在這門課程中，我們將教你如何")

