from dotenv import load_dotenv
import os
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

# Load environment variables from .env file

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize ChatGoogleGenerativeAI model

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Set up data folder
data_folder = p.cwd() / "data"
p(data_folder).mkdir(parents=True, exist_ok=True)

# Download and save PDF file
pdf_url = "https://athena.ecs.csus.edu/~buckley/CSc233/DevOps_for_Dummies.PDF"
pdf_file = str(p(data_folder, pdf_url.split("/")[-1]))
urllib.request.urlretrieve(pdf_url, pdf_file)

# Load PDF using PyPDFLoader
pdf_loader = PyPDFLoader(pdf_file)
pages = pdf_loader.load_and_split()


# Define a prompt template for question answering
prompt_template = """Supply a precise answer to the question using the context provided.Add 5 bullet points too. If the information needed is not evident in the context, indicate "answer not found in context" \n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:
                  """


# Create a PromptTemplate instance
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# # Load question answering chain
stuff_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Initialize text splitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)

# Create texts by splitting the context
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)

# Initialize GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

from langchain.vectorstores import FAISS

# Create vector index using FAISS
vector_index = FAISS.from_texts(texts, embeddings)

# Specify a question for similarity search
question = "How cloud and devops work together?"
# question = "How can devops be used to solve New challenges?"

# Find documents similar to the question in the vector index
docs = vector_index.similarity_search(question)
# print(docs)

# Run the question answering chain on the identified documents
stuff_answer = stuff_chain({"input_documents": docs, "question": question}, return_only_outputs=True)

answer = stuff_answer["output_text"]

formatted_answer = answer.replace('\n', '\n- ')

# Print the answer
print("Question:")
print(question)
print("\nAnswer:")
print("- " + formatted_answer)