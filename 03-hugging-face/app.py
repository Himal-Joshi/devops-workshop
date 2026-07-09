from transformers import pipeline

pipe = pipeline("text-classification", model="Dmyadav2001/Sentimental-Analysis")
print(pipe("I love AI and I want to learn more"))
