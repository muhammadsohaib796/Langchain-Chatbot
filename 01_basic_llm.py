from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

# Read the API key
api_key = os.getenv("GOOGLE_API_KEY")


# 2. INITIALIZE GEMINI MODEL
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
    google_api_key=api_key
)

# 3. PROMPT TEMPLATE

# template = """
# You are a helpful AI assistant.

# Current user question:
# {topic}

# Answer the user in a helpful and clear way in short.
# """

# topic = input("Enter the topic: ")

# prompt = PromptTemplate.from_template(template)

# formatted_prompt = prompt.invoke({
#     "topic":topic
# })
# print("\nFormatted Prompt:")
# print(formatted_prompt)


# 5. OUTPUT PARSER
parser = StrOutputParser()


# chain = prompt|llm|parser

history = ""

memory_template = """
You are a helpful AI assistant.

Here is the conversation history:
{history}

Current user question:
{topic}

Use the conversation history to understand the context.

If the user's name was mentioned in the conversation history,
remember the user's name.

Answer the current user question.
"""

memory_prompt = PromptTemplate.from_template(memory_template)

memory_chain = memory_prompt | llm | parser

# 11. CHAT LOOP

print("\n======================================")
print("AI Chatbot Started")
print("Type 'exit' to stop the chatbot.")
print("======================================")


while True:

    # Take user input
    topic = input("\nYou: ")


    # Exit condition
    if topic.lower() == "exit":
        print("\nChatbot stopped.")
        break


     # 12. RUN MEMORY CHAIN
    response = memory_chain.invoke({
    "history":history,    
    "topic":topic
    })

    print("\nAI: ", response)

    # 14. UPDATE CONVERSATION HISTORY

    
    history += f"\nUser: {topic}\nAI: {response}"


    # 15. VIEW CURRENT MEMORY
    print("\n--- Conversation History ---")
    print(history)