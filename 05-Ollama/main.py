from langchain_ollama import ChatOllama 

model =  ChatOllama(model="qwen3.5:2b")

res = model.invoke("Hello,how are you?")

print(res.content)

# def main():
#     print("Hello from 05-ollama!")


# if __name__ == "__main__":
#     main()
