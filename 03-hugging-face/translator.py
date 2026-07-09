import os
from dotenv import load_dotenv
# Import Marian specific classes instead of Auto classes to fix the config error
from transformers import MarianTokenizer, MarianMTModel

# 1. Load environment variables from the .env file
load_dotenv()

# 2. Retrieve the Hugging Face token
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found. Please ensure it is set in your .env file.")

model_name = "Helsinki-NLP/opus-mt-mul-en"

# 3. Use MarianTokenizer and MarianMTModel directly
tokenizer_mul_en = MarianTokenizer.from_pretrained(model_name, token=hf_token)
model_mul_en = MarianMTModel.from_pretrained(model_name, token=hf_token)

nepali_text_en = "नमस्ते, हजुरलाई आज कस्तो छ? मलाई आशा छ कि तपाईले हगिङ फेस मोडलहरू प्रयोग गरेर राम्रो समय बिताउनुहुनेछ!"

# Prepare input for the translation model
inputs_mul_en = tokenizer_mul_en(nepali_text_en, return_tensors="pt")

# Generate the translation
translated_ids_mul_en = model_mul_en.generate(inputs_mul_en.input_ids)

# Decode the generated IDs back to text
english_translation_ne = tokenizer_mul_en.decode(
    translated_ids_mul_en[0], 
    skip_special_tokens=True
)

print(f"Original Nepali: {nepali_text_en}")
print(f"Translated English: {english_translation_ne}")