import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredAPIFileIOLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
import sys

print(sys.path)

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
texts = text_splitter.split_documents(documents)

url = os.environ.get('QDRANT_URL')
api_key = os.environ.get('QDRANT_API_KEY')
print(api_key, url)
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    api_key=api_key,
    collection_name="test_dev"
)

if qdrant:
    print("___Connected to Vector DB__")

print("Ingestion was completed!")
del qdrant
print("Resources Released")
