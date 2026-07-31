# ==========================================
# 1. Imports
# ==========================================
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


# ==========================================
# 2. Load Environment Variables
# ==========================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ==========================================
# 3. Initialize LLM
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# ==========================================
# 4. Output Parser
# ==========================================

parser = StrOutputParser()


# ==========================================
# 5. Prompts
# ==========================================

chat_template = """
You are a helpful AI assistant.

Conversation Summary:
{summary}

Recent Conversation:
{recent_messages}

Current User Question:
{question}

Use the conversation summary and recent conversation to understand the context.

Answer the current user question in a short and clear way.
"""

chat_prompt = PromptTemplate.from_template(chat_template)


summary_template = """
You are an AI assistant.

Previous Summary:
{summary}

Old Conversation:
{old_messages}

Update the summary using the old conversation.

Keep only important information useful for future conversations.

Ignore unnecessary details.

Return only the updated summary.
"""

summary_prompt = PromptTemplate.from_template(summary_template)


# ==========================================
# 6. Chains 
# ==========================================

chat_chain = chat_prompt | llm | parser

summary_chain = summary_prompt | llm | parser


# ==========================================
# 7. Memory
# ==========================================

history = []
summary = ""

MAX_MESSAGES = 6



# ==========================================
# 8. Fucntion
# ==========================================

def ask_question(question: str) -> str:
    global history, summary

    recent_messages = history[-MAX_MESSAGES:]
    recent_messages_text = "\n".join(recent_messages)

    old_messages = history[:-MAX_MESSAGES]
    old_messages_text = "\n".join(old_messages)

    if old_messages:
        summary = summary_chain.invoke({
            "summary": summary,
            "old_messages": old_messages_text
    })
    response = chat_chain.invoke({
        "summary": summary,
        "recent_messages": recent_messages_text,
        "question": question
    })

    history.append(f"User: {question}")
    history.append(f"AI: {response}")

    return response