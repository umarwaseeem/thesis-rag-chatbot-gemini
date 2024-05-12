import streamlit as st
import pickle
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os

# Load the vector database
with open("vectorstore.pkl", "rb") as f:
    my_vector_database = pickle.load(f)

retriever = my_vector_database.as_retriever(search_kwargs={"k": 5})

# Define the template for the prompt
template = """
You are a helpful AI assistant.
Answer based on the context provided. 
context: {context}
input: {input}
answer:
"""

prompt = PromptTemplate.from_template(template)

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"])

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


def generate_response(query):
    response = retrieval_chain.invoke({"input": query})
    return response.get("answer")

# Set up Streamlit UI
st.title("Masters Thesis RAG Chatbot")
st.text("Project done by Muhammad Umar Waseem - i200762")

# Add input box for user queries
user_query = st.text_input("You:", "")

# Respond to user query
if st.button("Ask"):
    if user_query:
        answer = generate_response(user_query)
        st.write("AI:", answer)
    else:
        st.write("Please enter a question.")