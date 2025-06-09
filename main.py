from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# للسماح بالوصول من أي دومين (يفيد الباك اند بتاعك)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# إعداد Google Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

# تعريف الرسالة المستقبلة
class Message(BaseModel):
    message: str

@app.post("/chat")
async def chat(data: Message):
    user_input = data.message
    response = model.generate_content(user_input)
    return {"response": response.text}
