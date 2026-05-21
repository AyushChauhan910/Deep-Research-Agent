# Deep Research Agent

A deep research agent built with Groq (Llama 3.3 70B), Tavily search, and Streamlit UI.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # fill in your API keys
streamlit run app.py
```

## Agent Flow

Plan → Search → Fetch → Select Context → Synthesize → Critic
