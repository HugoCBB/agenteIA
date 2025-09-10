import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = str(os.getenv("GOOGLE_API_KEY")) 

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    api_key=GOOGLE_API_KEY
)

resp_text = llm.invoke("gere poemas romanticos")
print(resp_text)