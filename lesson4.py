from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages
)


import json
import dotenv
import os

# завантажити api ключі з папки .env
dotenv.load_dotenv()

# отримати сам ключ
api_key = os.getenv('GEMINI_API_KEY')

# створення чат моделі
# Велика мовна модель(llm)

llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',  # назва моделі
    google_api_key=api_key,    # ваша API
)

# response = llm.invoke("Привіт")
#
# print(type(response))
# print(response)
# print(repr(response))


# # історія спілкування(чат)
# messages = [
#     SystemMessage("""
#     Ти ввічливий чат бот. Твоя заача давати відповіді на питання
#     користувача.
#
#     Закінчуй усі відповіді фразою "Чи залишились іще питання?"
#     """),
#     HumanMessage('Привіт'),
#     AIMessage('Привіт, чим можу допомогти? Чи залишились іще питання?'),
#     HumanMessage('Яка столиця Франції?'),
#     AIMessage('Париж. Чи залишились іще питання?'),
#     HumanMessage("Розкажи пару фактів про це місто")
# ]
#
# response = llm.invoke(messages)
#
# print(response.content)

# простий чат бот

# messages = [
#     SystemMessage("""
#     Ти ввічливий чат бот. Твоя заача давати відповіді на питання
#     користувача.
#
#     Закінчуй усі відповіді фразою "Чи залишились іще питання?"
#     """)
# ]
#
# # основний цикл
# while True:
#     # отримати повідомлення від користувача
#     user_text = input('Ви: ')
#
#     # умова зупинки
#     if user_text == '':
#         break
#
#     # змінити тип даних на HumanMessage
#     human_message = HumanMessage(user_text)
#
#     # змінити історію
#     messages.append(human_message)
#
#     # застосувати модель
#     response = llm.invoke(messages)
#     # response -- AIMessage
#
#     # змінити історію
#     messages.append(response)
#
#     # вивід результату
#     print(f'AI: {response.content}')

# очищення історії повідомлень

# створення трімер
trimmer = trim_messages(
    strategy='last',  # залишати останні повідомлення

    token_counter=len,  # рахуємо кількість повідомлень
    max_tokens=5,  # залишати максимум 5 повідомлення(System, AI, Human)

    start_on='human',  # історія завжди починатиметься з HumanMessage
    end_on='human',  # історія завжди закінчуватиметься з HumanMessage
    include_system=True  # SystemMessage не чіпати
)

messages = [
    SystemMessage("""
    Ти ввічливий чат бот. Твоя заача давати відповіді на питання
    користувача.

    Закінчуй усі відповіді фразою "Чи залишились іще питання?"
    """)
]

# # основний цикл
# while True:
#     # отримати повідомлення від користувача
#     user_text = input('Ви: ')
#
#     # умова зупинки
#     if user_text == '':
#         break
#
#     # змінити тип даних на HumanMessage
#     human_message = HumanMessage(user_text)
#
#     # змінити історію
#     messages.append(human_message)
#
#     # почистити історію
#     messages = trimmer.invoke(messages)
#
#     # застосувати модель
#     response = llm.invoke(messages)
#     # response -- AIMessage
#
#     # змінити історію
#     messages.append(response)
#
#     # вивід результату
#     print(f'AI: {response.content}')
#
#     # вивід історії
#     print('\nІсторія')
#     for message in messages:
#         print(repr(message))
#     print()


# теж саме через ланцюг
chain = trimmer | llm

while True:
    # отримати повідомлення від користувача
    user_text = input('Ви: ')

    # умова зупинки
    if user_text == '':
        break

    # змінити тип даних на HumanMessage
    human_message = HumanMessage(user_text)

    # змінити історію
    messages.append(human_message)

    # застосувати модель
    response = chain.invoke(messages)
    # response -- AIMessage

    # змінити історію
    messages.append(response)

    # вивід результату
    print(f'AI: {response.content}')

    # вивід історії
    print('\nІсторія')
    for message in messages:
        print(repr(message))
    print()