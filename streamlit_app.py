# streamlit_app.py

import streamlit as st
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



def main():
    st.markdown("""
        <style>
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f0f2f5;
            color: #333;
        }

        .stApp {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            width: 90%;
            max-width: 800px;
            margin: auto;
        }

        h1 {
            color: #007bff;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }

        .css-1cpxqw2 {
            display: block;
            margin-top: 20px;
            font-weight: bold;
        }

        .stTextInput, .stNumberInput, .stSelectbox, .stFileUploader {
            width: 100%;
            padding: 12px;
            margin-top: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-sizing: border-box;
            font-size: 1em;
        }

        .stButton button {
            background: #007bff;
            color: white;
            padding: 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 20px;
            font-size: 1em;
            width: 100%;
        }

        .stButton button:hover {
            background: #0056b3;
        }

        .summaries {
            margin-top: 30px;
            background: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            max-height: 300px;
            overflow-y: auto;
        }

        .summary p {
            padding: 10px;
            border-bottom: 1px solid #eee;
        }

        .summary p:last-child {
            border-bottom: none;
        }

        .answers {
            margin-top: 20px;
        }

        .answer {
            background: #e9ecef;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Document Summarizer")

    st.sidebar.header("Configuration")
    chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=10000, step=100, value=2000)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=100, max_value=5000, step=100, value=200)
    chain_type = st.sidebar.selectbox("Chain Type", ["map_reduce", "stuff", "refine"])
    temperature = st.sidebar.number_input("Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
    num_summaries = st.sidebar.number_input("Number of Summaries", min_value=1, max_value=10, step=1, value=1)
    model_name = st.sidebar.selectbox("Model Name", ["GPT-3.5", "GPT-4", "GPT-4 Turbo"])

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    custom_prompt = st.text_input("Custom Prompt", value="Summarize the following document")

    if st.button("Summarize"):
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp.write(uploaded_file.read())
                pdf_file_path = temp.name

            try:
                if model_name == "GPT-3.5":
                    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, openai_api_key=openai_api_key)
                elif model_name == "GPT-4":
                    llm = ChatOpenAI(model_name="gpt-4", temperature=temperature, openai_api_key=openai_api_key)
                elif model_name == "GPT-4 Turbo":
                    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=temperature, openai_api_key=openai_api_key)
                else:
                    llm = ChatOpenAI(temperature=temperature, openai_api_key=openai_api_key)

                docs = setup_documents(pdf_file_path, chunk_size, chunk_overlap)
                doc_content = " ".join([doc.page_content for doc in docs])
                memory.save_context({"input": "Document content"}, {"output": doc_content})
                summaries = custom_summary(docs, llm, custom_prompt, chain_type, num_summaries)
                st.markdown("<div class='summaries'>Summaries:</div>", unsafe_allow_html=True)
                for summary in summaries:
                    st.markdown(f"<div class='summary'><p>{summary}</p></div>", unsafe_allow_html=True)
                st.session_state.summaries = summaries
                st.session_state.doc_content = doc_content
                st.session_state.show_ask = True
            finally:
                os.remove(pdf_file_path)

    if "show_ask" not in st.session_state:
        st.session_state.show_ask = False

    if st.session_state.show_ask:
        st.header("Ask a Question")
        question = st.text_input("Question:")
        if st.button("Ask"):
            if question:
                question_with_context = f"Document content: {st.session_state.doc_content}\n\nQuestion: {question}"
                llm = ChatOpenAI(temperature=0.0, openai_api_key=openai_api_key)
                conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
                answer = conversation.predict(input=question_with_context)
                st.markdown("<div class='answers'>Answers:</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='answer'><p>{answer}</p></div>", unsafe_allow_html=True)
                st.session_state.answers = st.session_state.answers + [answer] if "answers" in st.session_state else [answer]

if __name__ == "__main__":
    main()



# def main():
#     st.markdown("""
#         <style>
#         body {
#             font-family: 'Roboto', sans-serif;
#             background-color: #f0f2f5;
#             color: #333;
#         }

#         .stApp {
#             background: white;
#             padding: 40px;
#             border-radius: 10px;
#             box-shadow: 0 4px 8px rgba(0,0,0,0.1);
#             width: 90%;
#             max-width: 800px;
#             margin: auto;
#         }

#         h1 {
#             color: #007bff;
#             text-align: center;
#             margin-bottom: 30px;
#             font-size: 2.5em;
#         }

#         .css-1cpxqw2 {
#             display: block;
#             margin-top: 20px;
#             font-weight: bold;
#         }

#         .stTextInput, .stNumberInput, .stSelectbox, .stFileUploader {
#             width: 100%;
#             padding: 12px;
#             margin-top: 10px;
#             border: 1px solid #ccc;
#             border-radius: 5px;
#             box-sizing: border-box;
#             font-size: 1em;
#         }

#         .stButton button {
#             background: #007bff;
#             color: white;
#             padding: 15px;
#             border: none;
#             border-radius: 5px;
#             cursor: pointer;
#             margin-top: 20px;
#             font-size: 1em;
#             width: 100%;
#         }

#         .stButton button:hover {
#             background: #0056b3;
#         }

#         .summaries {
#             margin-top: 30px;
#             background: white;
#             padding: 20px;
#             border-radius: 5px;
#             box-shadow: 0 2px 5px rgba(0,0,0,0.1);
#             max-height: 300px;
#             overflow-y: auto;
#         }

#         .summary p {
#             padding: 10px;
#             border-bottom: 1px solid #eee;
#         }

#         .summary p:last-child {
#             border-bottom: none;
#         }

#         .answers {
#             margin-top: 20px;
#         }

#         .answer {
#             background: #e9ecef;
#             padding: 10px;
#             border-radius: 5px;
#             margin-top: 10px;
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     st.title("Document Summarizer")

#     st.sidebar.header("Configuration")
#     chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=10000, step=100, value=2000)
#     chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=100, max_value=5000, step=100, value=200)
#     chain_type = st.sidebar.selectbox("Chain Type", ["map_reduce", "stuff", "refine"])
#     temperature = st.sidebar.number_input("Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
#     num_summaries = st.sidebar.number_input("Number of Summaries", min_value=1, max_value=10, step=1, value=1)
#     model_name = st.sidebar.selectbox("Model Name", ["GPT-3.5", "GPT-4", "GPT-4 Turbo"])

#     uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
#     custom_prompt = st.text_input("Custom Prompt", value="Summarize the following document")

#     if st.button("Summarize"):
#         if uploaded_file is not None:
#             with tempfile.NamedTemporaryFile(delete=False) as temp:
#                 temp.write(uploaded_file.read())
#                 pdf_file_path = temp.name

#             try:
#                 if model_name == "GPT-3.5":
#                     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, openai_api_key=openai_api_key)
#                 elif model_name == "GPT-4":
#                     llm = ChatOpenAI(model_name="gpt-4", temperature=temperature, openai_api_key=openai_api_key)
#                 elif model_name == "GPT-4 Turbo":
#                     llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=temperature, openai_api_key=openai_api_key)
#                 else:
#                     llm = ChatOpenAI(temperature=temperature, openai_api_key=openai_api_key)

#                 docs = setup_documents(pdf_file_path, chunk_size, chunk_overlap)
#                 doc_content = " ".join([doc.page_content for doc in docs])
#                 memory.save_context({"input": "Document content"}, {"output": doc_content})
#                 summaries = custom_summary(docs, llm, custom_prompt, chain_type, num_summaries)
#                 st.markdown("<div class='summaries'>Summaries:</div>", unsafe_allow_html=True)
#                 for summary in summaries:
#                     st.markdown(f"<div class='summary'><p>{summary}</p></div>", unsafe_allow_html=True)
#                 st.session_state.summaries = summaries
#                 st.session_state.show_ask = True
#             finally:
#                 os.remove(pdf_file_path)

#     if "show_ask" not in st.session_state:
#         st.session_state.show_ask = False

#     if st.session_state.show_ask:
#         st.header("Ask a Question")
#         question = st.text_input("Question:")
#         if st.button("Ask"):
#             if question:
#                 doc_content = " ".join([doc.page_content for doc in setup_documents(pdf_file_path, chunk_size, chunk_overlap)])
#                 question_with_context = f"Document content: {doc_content}\n\nQuestion: {question}"
#                 llm = ChatOpenAI(temperature=0.0, openai_api_key=openai_api_key)
#                 conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
#                 answer = conversation.predict(input=question_with_context)
#                 st.markdown("<div class='answers'>Answers:</div>", unsafe_allow_html=True)
#                 st.markdown(f"<div class='answer'><p>{answer}</p></div>", unsafe_allow_html=True)
#                 st.session_state.answers = st.session_state.answers + [answer] if "answers" in st.session_state else [answer]

# if __name__ == "__main__":
#     main()

