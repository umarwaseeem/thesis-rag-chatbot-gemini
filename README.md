# Masters Thesis Retreival Augmented Generation Chatbot

### Author: [Muhammad Umar Waseem](https://github.com/Umar-Waseem)

## Contents


   - 1.1 [API Key](https://github.com/Umar-Waseem/thesis-rag-chatbot-gemini?tab=readme-ov-file#11-api-key)
   - 1.2 [Data Preprocessing](https://github.com/Umar-Waseem/thesis-rag-chatbot-gemini?tab=readme-ov-file#12-data-preprocessing)
   - 1.3 [Model Used](https://github.com/Umar-Waseem/thesis-rag-chatbot-gemini?tab=readme-ov-file#13-model-used)
   - 1.4 [Vector Store](https://github.com/Umar-Waseem/thesis-rag-chatbot-gemini?tab=readme-ov-file#14-vector-store)
   - 1.5 [Langchain Prompt Template](https://github.com/Umar-Waseem/thesis-rag-chatbot-gemini?tab=readme-ov-file#15-langchain-prompt-template)
   - 1.6 [Retreival Chain](https://github.com/Umar-Waseem/thesis-rag-chatbot-gemini?tab=readme-ov-file#16-retreival-chain)
   - 1.7 [Query User Interface](https://github.com/Umar-Waseem/thesis-rag-chatbot-gemini?tab=readme-ov-file#17-query-user-interface)


## 1 Langchain Based Retreival Augmented Generation

A langchain based RAG app has been made which works using vector embeddings and
google Gemini Pro LLM model.

### 1.1 API Key
```bash
export GOOGLE_API_KEY="your api key"
```

### 1.2 Data Preprocessing

**UnstructuredExcelLoader** has been used from langchain.document-loaders which is used
to load excel spreadsheet data into documents.

### 1.3 Model Used

Google Gemini Pro model has been used for getting contextual chat completion.

```python
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"])
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
```

### 1.4 Vector Store

FAISS (Facebook AI Similarity Search) Vector Store has been used to create and store
semantic embeddings for the loaded documents. The vector store can then later be queried
with a similarity search to get most relevant information.

```python
vectordb = FAISS.from_documents(documents=docs ,embedding=embeddings)

# save db as pickle file
with open("vectorstore.pkl", "wb") as f:
pickle.dump(vectordb, f)

#load db from pickle file
with open("vectorstore.pkl", "rb") as f:
my_vector_database = pickle.load(f)

# get 5 most relevant similar results
retriever = my_vector_database.as_retriever(search_kwargs={"k": 5})
```


### 1.5 Langchain Prompt Template

PromptTemplate has been used from langchain to craft efficient prompts which would later
be passed on to the model. The prompt also contains input variables which indicate to the
model that some information will be passed in by the user.

```python
template = """
You are a very helpful AI assistant.
You answer every question and apologize polietly if you dont know the answer.
The context contains information about a person,
title of their thesis,
the abstract of their thesis
and a link to their thesis.
Your task is to answer based on that information.
context: {context}
input: {input}
answer:
"""

prompt = PromptTemplate.from_template(template)
```

### 1.6 Retreival Chain

Retreival Chain has been used to pass documents/embeddings to the model as context for
Retreival Augmented Generation.

```python
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
```

### 1.7 Query User Interface

<img src="https://github.com/Umar-Waseem/thesis-rag-chatbot-gemini/blob/main/screenshot_streamlit.png" />



