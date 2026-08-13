# LLM
# Large Language Model
# велика мовна модель
# завантеження api key як змінну середовища
import dotenv
import os

import langchain
from langchain_google_genai import GoogleGenerativeAI
import langchain_google_genai

# завантадити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
# print(api_key)


# # модель
llm = GoogleGenerativeAI(
    model="gemini-3.6-flash",   # назва моделі
    api_key=api_key    # ключ до сервера з моделлю
)


# # використання
# response = llm.invoke("Привіт")
# print(response)


# параметри генерації
llm = GoogleGenerativeAI(
    model="gemini-3.6-flash",   # назва моделі
    api_key=api_key,    # ключ до сервера з моделлю
    temperature=1.9,    # температура
    top_p=0.8,
    top_k=10
)

response = llm.invoke("Придумай коротку історії про ельфа(до 5 речень)")
print(response)