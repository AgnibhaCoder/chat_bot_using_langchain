from flask import Flask, request, jsonify, send_from_directory
from flask_restful import Api, Resource
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import dotenv
from huggingface_hub import InferenceClient
import traceback
import logging

# Setting up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

dotenv.load_dotenv()


app = Flask(__name__)
api = Api(app)

@app.route('/')
def serve_index():
    return send_from_directory(os.getcwd(), 'index.html')

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()
except Exception as e:
    print(f"Error loading vector store: {e}")
    exit()

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

try:
    if not HF_ACCESS_TOKEN:
        raise ValueError("Hugging Face access token is missing.")
    client = InferenceClient(token=HF_ACCESS_TOKEN,model="distilgpt2")
except ValueError as e:
    print(e)
    exit()
except Exception as e:
    print(f"Error initializing Hugging Face client: {e}")
    exit()


class ChatBot(Resource):
    def post(self):
        try:
            data = request.get_json()
            query = data.get("question", "")
            if not query:
                logger.warning("Received empty question")
                return {"error": "Empty question"}, 400
            
            logger.info(f"Received query: {query}")

            docs = retriever.invoke(query)  # Keeping the retrieval
            context = "".join([d.page_content for d in docs])[:1000]  # Keeping the context creation
            logger.debug(f"Retrieved {len(docs)} documents, Context length: {len(context)}")

            prompt = f"Question: {query}\nContext: {context}" # Keeping the prompt formatting
            logger.info(f"Generated prompt for HF API: {prompt}")

            print("Prompt sent to HF:", prompt)

            response = client.text_generation(prompt=prompt)  # Direct call to HF API
            logger.info("Received response from Hugging Face API")

            if isinstance(response, str):
                return {"response": response}, 200
            else:
                logger.error("Unexpected response type from Hugging Face API")
                return {"error": "Unexpected response type"}, 500

        except Exception as e:
            logger.error(f"Error in ChatBot API: {e}", exc_info=True)
            return {"error": str(e)}, 500

api.add_resource(ChatBot, "/chat")

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == "__main__":
     app.run(host="0.0.0.0", port=5000)