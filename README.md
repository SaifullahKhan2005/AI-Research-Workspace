# AI-Research-Workspace

An AI-powered research workspace built with Python and Streamlit. It allows users to upload technical documentation query them using Retrieval-Augmented Generation (RAG).

🚀 Features
Document Ingestion: Supports automated loading and parsing of PDF and TXT files.
Semantic Search: Utilizes langchain-chroma for efficient local vector storage and retrieval.
Chunking: Implements recursive character splitting with configurable chunk sizes and overlaps for optimal context window management.
Gemini LLM Integration: Powered by Google's gemini-2.5-flash for high-speed, accurate synthesis of queried context.

🛠️ Tech Stack
Frontend: Streamlit
Orchestration: LangChain
Vector Database: ChromaDB
Embeddings & LLM: Google Generative AI (gemini-embedding-0010, gemini-2.5-flash)

Install Requirments:
  pip install -r requirements.txt

Configure API key:
  GOOGLE_API_KEY="your_api_key"

Run:
  streamlit run app.py

To run ensure that you add the path to your API key file inside the 'main_file.py" file. 
load_dotenv("path/filename"). Replace text inside function the API key path.

