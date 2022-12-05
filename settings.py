import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, 'media')
EARTHSAT_USER = os.getenv('EARTHSATUSER')
EARTHSAT_PASSWORD = os.getenv('EARTHSATPASSWORD')