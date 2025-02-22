import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_PATH = os.getcwd()
PROMPTS_DIR = os.path.join(BASE_PATH, r"data\prompts")

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
if not os.environ["GROQ_API_KEY"]:
    raise ValueError("GROQ_API_KEY environment variable is not set")
