from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages
)


import json
import dotenv
import os
from uuid import uuid4

# завантажити api ключі з папки .env
dotenv.load_dotenv()

# отримати сам ключ
api_key = os.getenv('GEMINI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# створення чат моделі
# Велика мовна модель(llm)

llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',  # назва моделі
    google_api_key=api_key,    # ваша API
)

# створення моделі для кодування текстів
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  # назва моделі
    google_api_key=api_key
)

# створення векторної бази даних
pc = Pinecone(api_key=pinecone_api_key)

# назва таблиці з документами
index_name = "itstep"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,   # назва таблиці
        dimension=768,     # кількість чисел у векторі
        metric="cosine",   # формула для обрахунку схожості
        spec=ServerlessSpec(
            cloud="aws",         # хмарна платформа(Амазон)
            region="us-east-1"   # регіон де знаходиться сервер(впливає на оплату)
        )
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)


def get_documents(query: str):
    """
    Шукає потрібні документи у вукторній базі даних.
    База містить таку інформацію:
    * факти про суп
    * факти про здоров'я

    :param query: str, запит від користувача
    :return: list[Document], список документів
    """
    docs = vector_store.similarity_search(
        query,
        k=2
    )
    print('hello from get_documents')
    return docs


messages = {"messages": [
    SystemMessage("""
    Ти ввічливий чат бот
    """)
    ]
}

agent = create_react_agent(
    model=llm,  # модель(чат модель)
    tools=[get_documents]  # список з інструментами
)
#messages = agent.invoke(messages)

while True:
    # отримати повідомлення від користувача
    user_text = input('Ви: ')

    # умова зупинки
    if user_text == '':
        break

    # змінити тип даних на HumanMessage
    human_message = HumanMessage(user_text)

    # змінити історію
    messages['messages'].append(human_message)

    # застосувати модель
    messages = agent.invoke(messages)

    # агент повертає всю історію повідомлень

    # отримати останнє повідомлення(відповідь моделі)
    ai_response = messages['messages'][-1]

    # вивід результату
    print(f"AI: {ai_response.content}")

    # вивід всієї історії
    print('\nІсторія')
    for message in messages['messages']:
        print(repr(message))
    print()
