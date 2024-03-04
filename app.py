import uvicorn
# from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import os
import json

app = FastAPI()

templates = Jinja2Templates(directory='templates')
app.mount("/static",StaticFiles(directory="static"),name="static")
local_llm = "ggml-model-Q4_K_M.gguf"

llm = LlamaCpp(model_path= "D:\Projects_2023\Flask_OpenAI_App\ggml-model-Q4_K_M.gguf",temperature=0.3,max_tokens=2048,top_p=1)

print("_____LLM Initialized____")
# Adding a Context for the LLM
prompt_template = """
Use the information provided to answer the questions.If you are unaware of the answer please notify the user that you don't know or need more information to be trained, don't make up any answer by yourself.
Context:{context}
Question:{question}
Only return a helpful and factually correct answer. The answer should be closely relevant and well explained.
Helpful and relevant  answer for the question.
"""

embedding = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
# Added Qdrant Cloud Cluster REST URI and Token
url = os.environ.get('QDRANT_URL')
api_key = os.environ.get('QDRANT_API_KEY')
print(api_key,url)
# url = "https://14bb26fd-e1d0-43fb-bdd2-ca46d8f6615d.us-east4-0.gcp.cloud.qdrant.io"
# api_key = "HNA-S5eVOclddg6HLLYUSFZAt05kJsAh4D19UwMcsb_XfQffpFiwJw"

client = QdrantClient(url=url,api_key=api_key,prefer_grpc=False)
db = Qdrant(client=client,embeddings = embedding,collection_name="test_dev")

prompt = PromptTemplate(template=prompt_template,input_variables=['context','question'])
retreiver = db.as_retriever(search_kwargs={"k":1})
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retreiver, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
    response = qa(query)
    print(response)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))
    
    res = Response(response_data)
    return res


if "__main__" == __name__:
    uvicorn.run(app, host="127.0.0.1", port=8001)

