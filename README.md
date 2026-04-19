<div align="center">
  <h1>☀️ Solar Power Forecasting & AI Grid Optimization Agent</h1>
  <p><strong>A Supervised Learning & Agentic Workflow Project</strong></p>
</div>

<br>

This project bridges the gap between traditional Machine Learning (Supervised Learning) and modern Agentic AI (Large Language Models + RAG). It predicts solar power generation (`DC_POWER`) based on environmental factors, and then uses a specialized AI agent to analyze the forecast, retrieve grid rules, and provide actionable recommendations.

---

## 🛠 Everything We Are Using (The Tech Stack)

### **Machine Learning & Data**
- **Scikit-Learn (Random Forest Regressor):** Chosen as the core prediction engine. Tree-based models are excellent for capturing the non-linear "bell curve" of solar generation (which peaks at noon and drops to zero at night) without requiring complex neural networks.
- **Pandas & NumPy:** For heavy data manipulation, feature extraction (datetime extraction), and vectorized mathematical operations.
- **Joblib & LZMA:** For serializing (pickling) the machine learning model. We use LZMA compression to shrink a massive 500MB+ Random Forest down to 82MB so it fits within GitHub limits for cloud deployment.

### **Agentic AI & Retrieval-Augmented Generation (RAG)**
- **Groq API (Llama-3.1-8b & Llama-3.3-70b):** Ultra-fast inference engines for our LLM. We use the 70B model for deep reasoning (evaluating grid risk) and the 8B model for our interactive chatbot.
- **ChromaDB:** A fast, local Vector Database used to store and query our solar grid rules. 
- **Sentence-Transformers (`all-MiniLM-L6-v2`):** Used to convert sentences (our documents and our queries) into high-dimensional numerical vectors. This allows ChromaDB to find "semantically similar" rules even if the exact keywords don't match.

### **Frontend & Deployment**
- **Streamlit:** Our web application framework. It allows us to build complex, interactive UI dashboards in pure Python.
- **Streamlit Community Cloud:** Used for continuous deployment.

---

## 🧠 Why & How We Are Using Specific Models

### 1. The Machine Learning Model (Random Forest)
**Why Random Forest?** 
Solar power generation relies heavily on Irradiation and Time of Day. Random Forests are an ensemble of decision trees. They split data based on rules (e.g., *Is Irradiation > 0.5? Is Hour > 18?*). This is uniquely perfectly suited for tracking the sun because the sun functionally turns off at night (0 irradiance)—a hard boundary that tree models grab instantly, whereas linear regression might struggle.
- **How we use it:** We train it on historical arrays of features (`[Ambient Temp, Module Temp, Irradiation, Hour, Month]`) to output a single continuous variable (`DC_POWER`).

### 2. The Vector Embedding Model (`all-MiniLM-L6-v2`)
**Why this model?**
It is a lightweight sentence transformer that maps text to a 384-dimensional dense vector space. 
- **How we use it:** When the Agent sees a high power forecast, it needs to know the grid rules. We use this model to encode both the grid rules (`data/docs/grid_rules.txt`) into ChromaDB, and later encode the Agent's query into a vector to find the closest matching rule by calculating cosine similarity.

### 3. The Large Language Models (LLaMA via Groq)
**Why LLaMA?**
Open-source frontier models. 
- **How we use it:** We restrict the LLM tightly using System Prompts so it behaves strictly as a Grid Optimization Expert. It receives the ML prediction and the RAG document injection, synthesizes both, and returns JSON-formatted risk profiles (Low/Medium/High).

---

## 🎓 Concepts Applied

1. **Supervised Learning (Regression):** Predicting a continuous numerical output based on labeled training data.
2. **Feature Engineering:** Extracting `hour` and `month` from raw datetime strings because the model cannot read time linearly (seasonality and daily cadence matter).
3. **Retrieval-Augmented Generation (RAG):** Instead of fine-tuning an LLM on grid rules, we use ChromaDB to retrieve the documents in real-time and paste them into the LLM's prompt window.
4. **Agentic Workflows:** The code dynamically executes steps: Forecast -> Summarize -> Query DB -> Prompt LLM -> Parse Response safely.
5. **Gini Feature Importance:** Trees evaluate splits based on "Gini impurity" reduction, giving us automatic Explainability on which feature drove the forecast (Irradiation dominates).
6. **Bias-Variance Tradeoff:** Using Cross-Validation and Holdout sets to ensure our model generalizes to future days rather than hyper-memorizing the past.

---

## 📖 Deep Code Walkthrough (Module by Module)

### `app/streamlit_app.py`
This is your "Main Loop." 
- `st.set_page_config` and the massive blob of `<style>` HTML establishes your premium, modern CSS (glassmorphism tabs, dark gradients).
- `@st.cache_resource` on `load_model()` prevents Streamlit from reloading an 80MB file every time a user clicks a button. It checks if the `lzma` (compressed) model exists, and natively decompresses it into RAM.
- **Tab Routing:** Divides the logic into `Predict`, `Data Analysis`, `Model Evaluation`, `Forecast`, and `Logs`.
- `st.bar_chart` and `matplotlib.pyplot` blocks visualize data distribution.

### `src/rag/retriever.py`
This module manages the Semantic memory of the app.
- `SentenceTransformer("all-MiniLM-L6-v2")`: Downloads the embedding weights.
- `chromadb.Client()`: Instantiates the database.
- `load_docs()`: Opens your `grid_rules.txt`, splits the text by line (chunking technique), encodes each line into vectors, and saves them to a collection.
- `query_docs(query)`: Takes the summarized solar forecast context, encodes it into a vector, and uses `collection.query()` to fetch the top 3 most relevant rules mathematically.

### `src/agent/solar_agent.py`
This is the core Brain orchestrator.
- `run_agent(predictions)`: This function chains everything together. 
  1. It summarizes the numeric predictions into text. 
  2. Runs `query_docs` to ask ChromaDB for context. 
  3. Passes the summary and RAG knowledge to `llm_reason()`.
- `llm_reason()`: Crafts a rigid system prompt demanding the LLM formats its output STRICTLY as JSON with specific keys (`risk`, `reasoning`, `recommendations`).
- `json.loads(llm_output)`: Tries to parse the LLM's string into a Python Dictionary. If the LLM hallucinates formatting, the `try-except` block falls back to a safe "Unknown" risk state to prevent the web app from crashing.

### `src/agent/chatbot.py`
- Exposes an interactive Q&A capability. 
- Includes rigid constraints explicitly telling the LLm: *"If the User Question is unrelated to solar power forecasting... you MUST refuse to answer."* This safeguards against hallucination and off-topic conversations (e.g. users asking for code snippets or recipes).

### `requirements.txt`
This dictates exactly what Streamlit Cloud must pip-install on its virtual machine to recreate your environment. 
- We explicitly install `opentelemetry` upgrades because `chromadb` has compatibility conflicts with Python 3.14/Protobuf 5.x.

### `.gitignore` 
- Critical for deployment. It tells GitHub to ignore `models/*.pkl` (the uncompressed heavy model) but uses an exclamation mark `!models/*.pkl.lzma` to explicitly permit the 80MB compressed model to push to the cloud safely.
