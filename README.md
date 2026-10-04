# Agentic Bots with Tools

A lightweight LangGraph-based AI agent that combines a Gemini LLM with multiple tools for research, calculations, weather, stock data, and PDF-based retrieval. The project is designed to demonstrate how an agent can decide which tool to use based on the user's request and then respond with grounded results.

## What this project does

This agent can:

- answer general questions with an LLM
- perform arithmetic using a calculator tool
- fetch current weather for a city
- fetch stock quotes for a ticker symbol
- search the web with Tavily
- answer questions from a local PDF via RAG
- simulate a stock purchase flow that pauses for human approval

## Tech stack

- Python
- LangGraph
- LangChain
- Google Gemini
- Tavily Search
- FAISS
- PyPDF
- SQLite for chat checkpointing

## Project structure

- `backend.py` — agent definition, tools, graph setup, and demo run
- `requirements.txt` — Python dependencies
- `.env` — local environment variables (not committed)
- `chatbot.db` — SQLite chat history/checkpoint storage created at runtime
- `faiss-db/` — FAISS index generated from the uploaded PDF
- `science2.pdf` — example PDF used for retrieval

## Prerequisites

- Python 3.10+
- A Google API key for Gemini
- A Tavily API key for web search
- Optional: OpenWeather API key for weather tool
- Optional: Alpha Vantage API key for stock pricing tool

## Setup

1. Clone the repository and move into it:

   ```bash
   git clone https://github.com/sailesh730/Agenctic-bots-with-tools.git
   cd Agenctic-bots-with-tools
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your keys:

   ```env
   GOOGLE_API_KEY=your_google_gemini_key
   TAVILY_API_KEY=your_tavily_key
   OPENWEATHER_API_KEY=your_openweather_key
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   ```

   Note: the code accepts either `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

## Running the agent

From the project root:

```bash
python backend.py
```

The script runs a demo invocation:

```python
response = app.invoke(
    {"messages": [HumanMessage(content="1 share of apple and latest news of ai")]},
    config={"configurable": {"thread_id": "demo-thread"}},
)
```

This is a basic test of the graph and tool orchestration.

To launch the Streamlit chat interface, run:

```bash
python -m streamlit run frontend.py
```

## Docker setup

This project includes Docker support so you can run the agent in a container.

1. Copy the example environment file and update the values:

   ```bash
   copy .env.example .env
   ```

2. Build and run the Streamlit app:

   ```bash
   docker compose up --build
   ```

3. Open [http://localhost:8501](http://localhost:8501) in your browser.

4. To stop the container:

   ```bash
   docker compose down
   ```

The Docker setup uses the `Dockerfile` and `docker-compose.yml` files in the project root.

## How the RAG PDF tool works

The project includes a `rag_tool` that:

- loads a PDF with `PyPDFLoader`
- splits the content into chunks
- embeds those chunks with a sentence-transformer model
- stores them in a FAISS vector database
- retrieves relevant passages when the user asks document-based questions

To use the PDF search, ensure your document is saved in the root directory and the app can load it. The example file currently included is `science2.pdf`.

## Notes

- Chat history is stored in `chatbot.db` using `SqliteSaver`.
- The FAISS retrieval index is stored in the `faiss-db/` folder.
- The `purchase_stock` tool intentionally requires human approval via `interrupt(...)` before completing a simulated purchase.
- The project is intentionally a demo and can be extended into a web UI, more tools, or a more advanced multi-agent workflow.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
