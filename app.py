from flask import Flask, request, jsonify, send_from_directory
import logging
import sys
import os
from werkzeug.utils import secure_filename
import PyPDF2
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

llm = LlamaCPP(
    model_url='https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/resolve/main/zephyr-7b-alpha.Q5_K_M.gguf',
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    model_kwargs={"n_gpu_layers": -1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large")

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)
index = None
query_engine = None

@app.route('/')
def serve_frontend():
    return send_from_directory('', 'index.html')

@app.route('/index-pdf', methods=['POST'])
def index_pdf():
    global index
    if 'pdf_path' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['pdf_path']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Read the PDF file content
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text()

    # Create a Document from the text
    document = Document(text=pdf_text)
    documents = [document]

    global service_context
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    return jsonify({"message": "PDF indexed successfully"}), 200

@app.route('/get-answers', methods=['GET'])
def get_answers():
    global query_engine
    question = request.args.get('question')
    if index is None:
        return jsonify({"error": "Index not initialized. Please index PDFs first."}), 400
    else:
        query_engine = index.as_query_engine()
        response = query_engine.query(question)
        response_json = {"answer": str(response)}
        return jsonify(response_json), 200

if __name__ == '__main__':
    app.run(debug=True)
