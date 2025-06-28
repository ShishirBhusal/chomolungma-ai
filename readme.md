# Chomolungma AI - Your Multimodal AI Trekking Companion 🏔️
 
---

## 🚀 About The Project

Chomolungma AI is a full-stack, multimodal AI trekking guide for Nepal, designed to be a comprehensive companion for travelers. This project moves beyond simple Q&A bots to create a stateful, intelligent agent that assists users with planning **before** their trek and can provide information **during** their adventure.

Instead of spending hours searching blogs and forums, users can interact with a single, expert AI to get personalized, safe, and detailed plans. The agent is built with a strong emphasis on safety protocols and responsible trekking practices.

### Key Features:
*   **🤖 Stateful, Multi-Step Reasoning:** Built with LangGraph, the agent can execute complex plans, moving sequentially through steps like itinerary creation, gear list generation, and budgeting.
*   **🧠 Retrieval-Augmented Generation (RAG):** The agent's knowledge is grounded in a vector database (ChromaDB) containing specific information on trekking routes, costs, gear, and safety rules, ensuring factual and relevant responses.
*   **👁️ Multimodal Vision:** Leveraging Google's Gemini 2.5 flash, users can upload photos of mountains, gear, or landmarks and get intelligent, contextual identification and information.
*   **🔐 Secure User Authentication:** A full authentication system with a Supabase (PostgreSQL) backend allows users to register, log in, and have their conversation history securely saved.
*   **⚡ Real-time Streaming UI:** The frontend, built with Streamlit, provides a modern chat experience with real-time status updates (showing the agent's "thoughts") and streaming "typing" effects for AI responses.
*   **✅ Full-Stack Architecture:** A robust system with a clear separation of concerns between the frontend (Streamlit), backend API (FastAPI), and AI core logic.

---

## 🛠️ Tech Stack

This project utilizes a modern stack for building production-ready AI applications:

*   **AI Core & Orchestration:**
    *   [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/): For building the stateful, multi-step agent.
    *   [Google Gemini 1.5 Pro & Flash](https://deepmind.google/technologies/gemini/): As the core Large Language Models for generation, vision, and routing.
*   **Backend:**
    *   [FastAPI](https://fastapi.tiangolo.com/): For creating the robust, asynchronous API server.
    *   [Uvicorn](https://www.uvicorn.org/): As the ASGI server.
*   **Frontend:**
    *   [Streamlit](https://streamlit.io/): For building the interactive web application and chat interface.
*   **Database & RAG:**
    *   [Supabase](https://supabase.com/): As the PostgreSQL database for user and conversation data.
    *   [ChromaDB](https://www.trychroma.com/): As the vector store for the RAG system.
*   **Authentication:**
    *   Supabase Auth & JWT (JSON Web Tokens).

---

## ⚙️ Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   Python 3.10+
*   An account with [Supabase](https://supabase.com/) to create a database.
*   A [Google AI API Key](https://ai.google.dev/).

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ShishirBhusal/chomolungma-ai
    cd chomolungma-ai
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables:**
    *   Create a `.env` file in the root directory by copying the example:
        ```bash
        cp .env.example .env
        ```
    *   Open the `.env` file and add your secret keys:
        ```
        GOOGLE_API_KEY="your_google_api_key"
        SUPABASE_URL="https://your-project-url.supabase.co"
        SUPABASE_ANON_KEY="your_supabase_anon_key"
        ```

4.  **Set up the Supabase Database:**
    *   Create a new project on [Supabase](https://supabase.com/).
    *   Navigate to the **SQL Editor** and run the SQL script provided in `database_setup.sql` to create the necessary tables and security policies. *(You would need to create this file from our chat history)*.

5.  **Build the Knowledge Base:**
    *   Add your source documents (e.g., `.txt`, `.pdf` files about trekking) to the `knowledge_base_source_data` directory.
    *   Run the ingestion script to build the ChromaDB vector store:
        ```bash
        python create_knowledge_base.py
        ```
        *(This assumes you have a `create_knowledge_base.py` script)*

### Running the Application

You will need to run two services in separate terminals from the root directory.

1.  **Terminal 1: Start the FastAPI Backend Server:**
    ```bash
    uvicorn main:app --reload
    ```

2.  **Terminal 2: Start the Streamlit Frontend:**
    ```bash
    streamlit run app_ui.py
    ```

Open your browser to `http://localhost:8501` and you should see the Chomolungma AI login screen!

---
