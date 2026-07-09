from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")   
)

response = model.invoke("Tell me about Quantum Computing ")
print(response.content.text)