from transformers import pipeline

classifier = pipeline("sentiment-analysis")
a= classifier("I've been waiting for a HuggingFace course my whole life.")
print(a)

# zeroshot= pipeline("zero-shot-classification")
# zeroshotanswer=classifier(
#     "This is a course about the Transformers library",
#     candidate_labels=["education", "politics", "business"],
# )

# print(zeroshotanswer)