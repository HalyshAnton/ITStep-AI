# пошук потрібного документа
# RAG -- (пошук - відповідь - генерація)

# документ1 -- Суп корисний при застуді
# документ2 -- Суп придумали в Китаї
# документ3 -- Бігати більше 10 км шкідливо для здоров'я

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

import dotenv
import os
from uuid import uuid4

# завантажити api ключі з папки .env
dotenv.load_dotenv()

# отримати сам ключ
api_key = os.getenv('GEMINI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# створення моделі для кодування текстів
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  # назва моделі
    google_api_key=api_key
)

# # кодування тексту
# vec1 = embeddings.embed_query('Фільм чудовий')
# vec2 = embeddings.embed_query('Цей фільм чудовий')
# vec3 = embeddings.embed_query('Дуже хороший фільм')
#
# # закодовані числа(вектори)
# print(vec1)
# print(vec2)
# print(vec3)
#
# # кількість чисел у векторі
# print(len(vec1))  # 768

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

# створення документів
# документ1 -- Суп корисний при застуді
doc1 = Document(
    page_content="Суп корисний при застуді",  # вміст документа
    # мета дані(додаткова інформація)
    metadata={
        'type': "здоров'я",
        'author': "Anton Halysh"
    }
)

# документ2 -- Суп придумали в Китаї
doc2 = Document(
    page_content="Суп придумали в Китаї",  # вміст документа
    # мета дані(додаткова інформація)
    metadata={
        'type': "історія",
        'author': "Anton Halysh",
        'date': "2025"
    }
)

# документ3 -- Бігати більше 10 км шкідливо для здоров'я
doc3 = Document(
    page_content="Бігати більше 10 км шкідливо для здоров'я",  # вміст документа
    # мета дані(додаткова інформація)
    metadata={
        'type': "здоров'я",
        'author': "Anton Halysh"
    }
)

# добавляння документів
# створити ID для документів

docs = [doc1, doc2, doc3]
ids = [str(uuid4()) for _ in range(len(docs))]

#vector_store.add_documents(docs, ids=ids)

# як дістати потрібний документ
user_text = 'Розкажи щось цікаве про суп'

# пошук схожого документа
docs = vector_store.similarity_search(
    user_text,   # запит від користувача
    k=2,          # кількість документів
    # фільтр по метаданих
    filter={'type': "здоров'я"}  # документи про здоров'я
)

for doc in docs:
    print(doc)
