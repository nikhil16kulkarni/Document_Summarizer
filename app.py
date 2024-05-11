# app.py

from flask import Flask, request, render_template, jsonify
import openai
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)
openai_api_key = os.getenv("OPENAI_API_KEY", "your_actual_api_key")

memory = ConversationBufferMemory()

def setup_documents(pdf_file_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(pdf_file_path)
    docs_raw = loader.load()
    docs_raw_text = [doc.page_content for doc in docs_raw]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents(docs_raw_text)
    return docs

def custom_summary(docs, llm, custom_prompt, chain_type, num_summaries):
    custom_prompt = custom_prompt + """:\n {text}"""
    COMBINE_PROMPT = PromptTemplate(template=custom_prompt, input_variables=["text"])
    MAP_PROMPT = PromptTemplate(template="Summarize:\n{text}", input_variables=["text"])
    if chain_type == "map_reduce":
        chain = load_summarize_chain(llm, chain_type=chain_type, map_prompt=MAP_PROMPT, combine_prompt=COMBINE_PROMPT)
    else:
        chain = load_summarize_chain(llm, chain_type=chain_type)
    summaries = []
    for i in range(num_summaries):
        summary_output = chain.invoke({"input_documents": docs}, return_only_outputs=True)["output_text"]
        summaries.append(summary_output)
    return summaries


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    custom_prompt = request.form['custom_prompt']
    chunk_size = int(request.form['chunk_size'])
    chunk_overlap = int(request.form['chunk_overlap'])
    chain_type = request.form['chain_type']
    num_summaries = int(request.form['num_summaries'])
    temperature = float(request.form['temperature'])
    model_name = request.form['model_name']


    file = request.files['file']
    if file:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)
            pdf_file_path = temp.name

        try:
            if model_name == "ChatGPT":
                llm = ChatOpenAI(temperature=temperature, openai_api_key=openai_api_key)
            elif model_name == "GPT-4":
                llm = ChatOpenAI(model_name="gpt-4", temperature=temperature, openai_api_key=openai_api_key)
            else:
                llm = ChatOpenAI(temperature=temperature, openai_api_key=openai_api_key)

            docs = setup_documents(pdf_file_path, chunk_size, chunk_overlap)
            doc_content = " ".join([doc.page_content for doc in docs])
            memory.save_context({"input": "Document content"}, {"output": doc_content})
            summaries = custom_summary(docs, llm, custom_prompt, chain_type, num_summaries)
            return jsonify(summaries=summaries)
        finally:
            os.remove(pdf_file_path)
    return jsonify(summaries=[])

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    llm = ChatOpenAI(temperature=0.0, openai_api_key=openai_api_key)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    answer = conversation.predict(input=question)
    return jsonify(answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
