from transformers import pipeline
import opencc

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


def masker(text,top_k=2):
    masker = pipeline("fill-mask")
    maskeranswer = masker(text,top_k)
    print(maskeranswer)

def named_entity_recognition(text):
    ner = pipeline("ner",grouped_entities=True)
    ### per 人
    ### org 組織
    ### loc 地點
    ### misc 其他
    neranswer = ner(text)
    print(neranswer)

### 問答系統
def question_answering(context,question):
    qa = pipeline("question-answering")
    qaanswer = qa(context=context,question=question)
    print(qaanswer)

def summarization(text):
    summarizer = pipeline("summarization")
    summarizeranswer = summarizer(text)
    print(summarizeranswer)

def translation(text):
    translator = pipeline("translation",model="Helsinki-NLP/opus-mt-en-zh")
    translatoranswer = translator(text)
    # print(translatoranswer)
    converter = opencc.OpenCC('s2t')  # 簡體到繁體
    traditional_text = converter.convert(translatoranswer[0]['translation_text'])
    print(traditional_text)

# sentiment_analysis()
# zero_shot_classification()
# EnglishTextGenerator().generate_text("I've been waiting for a HuggingFace course my whole life.")
# ChineseTextGenerator().generate_text("在這門課程中，我們將教你如何")
# masker("This course will teach you all about <mask> models.")
# named_entity_recognition("My name is Wolfgang and I live in Berlin")
# question_answering(context="My name is Wolfgang and I live in Berlin",question="What is my name?")
# summarization("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. The tower is a global cultural icon of France and one of the most recognizable structures in the world.")
translation("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. The tower is a global cultural icon of France and one of the most recognizable structures in the world.")


