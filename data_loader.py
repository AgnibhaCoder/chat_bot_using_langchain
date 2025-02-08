import time
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


urls=["https://brainlox.com/courses/category/technical"]

time.sleep(2)
loader=UnstructuredURLLoader(urls)
documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs=text_splitter.split_documents(documents)

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_store=FAISS.from_documents(docs,embeddings)

vector_store.save_local("faiss_index")