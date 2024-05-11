# Document Summarizer

This project is a web-based application for summarizing PDF documents using advanced language models. The application is built with Streamlit and utilizes the Hugging Face Transformers library for model loading and inference. The summarizer supports custom prompts and allows users to ask questions about the document after summarizing it.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Features

- Upload PDF documents and generate summaries.
- Customizable chunk size and overlap for text splitting.
- Supports different summarization chain types (map_reduce, stuff, refine).
- Adjustable model temperature and number of summaries.
- Ask questions about the document content after summarization.
- Supports multiple OpenAI models (GPT-3.5, GPT-4, GPT-4 Turbo).

## Installation

### Prerequisites

- Python 3.7+
- Anaconda or virtual environment (recommended)

### Clone the Repository

```bash
git clone https://github.com/nikhil16kulkarni/document-summarizer.git
cd document-summarizer
```

### Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application Locally

1. Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY=your_openai_api_key  # On Windows: set OPENAI_API_KEY=your_openai_api_key
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and go to http://localhost:8501

### Using the Application on Hugging Face Spaces
You can also access the application hosted on Hugging Face Spaces - https://huggingface.co/spaces/nikhil16kulkarni/document-summarizer


## Configuration
The application allows users to configure various parameters through the sidebar:

**Chunk Size:** Size of text chunks for splitting the document. </br>
**Chunk Overlap:** Overlap size between text chunks. </br>
**Chain Type:** Type of summarization chain (map_reduce, stuff, refine). </br>
**Temperature:** Temperature setting for the model. </br>
**Number of Summaries:** Number of summaries to generate. </br>
**Model Name:** Select between GPT-3.5, GPT-4, and GPT-4 Turbo. </br>


## How It Works

**Document Upload:** Users can upload a PDF document which is then processed using PyPDFLoader to extract text content.

**Text Splitting:** The extracted text is split into chunks using RecursiveCharacterTextSplitter with configurable chunk size and overlap.

**Summarization:**

1. The text chunks are summarized using the selected language model and chain type.
2. A custom prompt can be provided to tailor the summarization output.

**Question Answering:**

1. After summarization, users can ask questions related to the document.
2. The document content is stored in a conversation buffer memory to provide context-aware answers.

**Code Overview**
1. app.py: The main Streamlit application file.
2. requirements.txt: Lists all the dependencies required for the project.

**Key Functions**
1. setup_documents(pdf_file_path, chunk_size, chunk_overlap): Loads and splits the document into chunks.
2. custom_summary(docs, llm, custom_prompt, chain_type, num_summaries): Generates summaries using the selected model and chain type.
3. main(): The main function that sets up the Streamlit interface and handles user interactions.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


## License

This project is licensed under the MIT License.
