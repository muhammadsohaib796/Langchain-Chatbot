from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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

template = """
You are a helpful AI assistant.

Current user question:
{topic}

Answer the user in a helpful and clear way in short.
"""

# topic = input("Enter the topic: ")

prompt = PromptTemplate.from_template(template)

# formatted_prompt = prompt.invoke({
#     "topic":topic
# })
# print("\nFormatted Prompt:")
# print(formatted_prompt)


parser = StrOutputParser()

chain = prompt|llm|parser




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

    response = chain.invoke({
    "topic":topic
    })

    print("\nAI Response:")
    print(response)