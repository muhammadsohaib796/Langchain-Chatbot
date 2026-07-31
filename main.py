from fastapi import FastAPI

from chatbot import ask_question
from models import ChatRequest, ChatResponse

app = FastAPI()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    response = ask_question(request.question)

    return ChatResponse(answer=response)