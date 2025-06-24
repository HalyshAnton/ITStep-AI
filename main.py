
import dotenv
import os


# завантажити api ключі з папки .env
dotenv.load_dotenv()

# отримати сам ключ
api_key = os.getenv('SERPER_API_KEY')

print(api_key)

