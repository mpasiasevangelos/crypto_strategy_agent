An agent using Langraph and Ollama to identify trading strategies. It's work in progress.

This project sets up an AI-powered trading assistant using LangChain, Ollama, FAISS, and technical indicators (MACD, EMA, SMA). 
It retrieves trading strategies from a FAISS vector store and generates trading insights based on OHLCV Bitcoin data.

This script loads OHLCV Bitcoin data from a CSV file, calculates technical indicators, and retrieves relevant trading strategies.
The graph workflow dynamically decides whether to retrieve, rewrite, or generate responses based on document relevance.
The LangChain + Ollama integration allows for customized LLM-based decision-making in trading.

