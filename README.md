# Multi-Agent Financial Portfolio Assistant

A sophisticated AI-powered system for analyzing Asian tech stock markets using a multi-agent architecture that combines real-time market data, sentiment analysis, quantitative metrics, and natural language processing.

## 🎯 Project Overview

This project implements a comprehensive financial analysis system using 6 specialized AI agents working together to provide intelligent portfolio insights through multiple interfaces including voice interaction.

**Coverage:** 19 tech stocks across 4 Asian regions (East Asia, South Asia, Southeast Asia, Western Asia)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Architecture](#-solution-architecture)
- [Key Features](#-key-features)
- [Agent Descriptions](#-agent-descriptions)
- [Technical Stack](#️-technical-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Challenges & Solutions](#️-challenges--solutions)
- [Future Enhancements](#-future-enhancements)

---

## 🔴 Problem Statement

### Core Challenges:

- **Fragmented Data Sources:** Market data scattered across multiple APIs and platforms
- **Manual Analysis Burden:** Time-consuming to track 19+ stocks across multiple regions
- **Sentiment Blindness:** Missing market sentiment signals from news and social media
- **Lack of Integration:** No unified system combining quantitative metrics with qualitative sentiment
- **Accessibility Gap:** Complex portfolio analysis not accessible through natural interfaces (voice, chat)
- **Real-time Decision Making:** Difficulty in getting actionable insights quickly for trading decisions
- **Regional Complexity:** Asian markets span multiple timezones, languages, and market dynamics
- **Information Overload:** Too much data, not enough actionable intelligence

### Target Users:

- Retail investors managing personal portfolios
- Portfolio managers overseeing multiple Asian tech stocks
- Financial analysts requiring comprehensive regional insights
- Traders needing quick sentiment + quantitative analysis

---

## 💡 Solution Architecture

### Multi-Agent System Design:

The solution implements 6 specialized agents working in a coordinated pipeline:

1. **API Agent** → Fetches real-time market data
2. **Scraping Agent** → Analyzes news sentiment
3. **Retriever Agent** → Enables semantic search (RAG)
4. **Analysis Agent** → Performs quantitative calculations
5. **Language Agent** → Generates natural language insights
6. **Voice Agent** → Provides hands-free interaction
7. **Orchestrator Agent** → Coordinates workflows

Each agent is autonomous, specialized, and outputs structured JSON for downstream consumption.

---

## ✨ Key Features

### Data Collection & Analysis:

- **Real-time Market Data:** Yahoo Finance API integration for prices, volumes, earnings
- **Sentiment Analysis:** Web scraping with Firecrawl/BeautifulSoup for news sentiment
- **Multi-source Fusion:** Combines quantitative + qualitative data
- **19 Stock Coverage:** Major tech companies across 4 Asian regions

### Intelligence Layer:

- **RAG Pipeline:** FAISS vector store + sentence transformers for semantic search
- **Quantitative Metrics:** Portfolio allocation, Sharpe ratio, volatility, VaR, HHI concentration
- **Earnings Analysis:** Identifies beats/misses with threshold-based alerts
- **Risk Assessment:** Multi-dimensional risk scoring with actionable insights

### Natural Language Processing:

- **LLM Integration:** Google Gemini 2.0 for natural language generation
- **Morning Briefs:** Automated comprehensive portfolio summaries
- **Q&A System:** Answer specific portfolio questions using RAG
- **Rate Limiting:** Intelligent caching + request queuing

### Voice Interface:

- **Speech Recognition:** Google Speech API with ambient noise calibration
- **Text-to-Speech:** gTTS for audio responses
- **Keyword Validation:** Ensures portfolio-related queries
- **Graceful Fallback:** Text input when recognition fails

### User Interfaces:

- **Streamlit Dashboard:** Visual analytics with interactive charts (Plotly)
- **AI Chatbot:** Text-based portfolio assistant
- **Voice Assistant:** Hands-free queries with audio recorder
- **Command-line Tools:** Batch processing for automation

---

## 🤖 Agent Descriptions

### 1. API Agent (`api_agent.py`)

**Purpose:** Fetch real-time market data from Yahoo Finance

**Key Functions:**
- Fetches current prices, previous close, change %, volume for 19 stocks
- Retrieves market cap, sector, P/E ratio, earnings data
- Supports regional filtering (East Asia, South Asia, etc.)
- Handles rate limiting (0.5s delay between requests)
- **Outputs:** `multi_region_results_TIMESTAMP.json`

**Technologies:**
- `yfinance` for Yahoo Finance API
- `numpy` for numerical calculations
- JSON persistence with NumPy type conversion

---

### 2. Scraping Agent (`scraping_agent.py`)

**Purpose:** Analyze market sentiment from news articles

**Key Functions:**
- Scrapes Google News RSS feeds for each stock
- Extracts headlines + full article content (Firecrawl API or BeautifulSoup fallback)
- Weighted sentiment scoring with positive/negative keywords
- Regional sentiment aggregation
- **Outputs:** `regional_sentiment_TIMESTAMP.json`

**Technologies:**
- `feedparser` for RSS parsing
- `BeautifulSoup` for HTML scraping
- Firecrawl MCP API for advanced scraping
- Custom sentiment analyzer with keyword weighting

---

### 3. Retriever Agent (`retriever_agent.py`)

**Purpose:** Implement RAG pipeline for semantic search

**Key Functions:**
- Loads data from API + Scraping agents
- Chunks documents (512 chars, 50 overlap)
- Generates embeddings using sentence-transformers
- Indexes in FAISS vector store
- Supports semantic queries like "stocks with high growth"
- **Outputs:** `vector_store/faiss_index.bin` + `metadata.json`

**Technologies:**
- FAISS for vector similarity search
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- `langchain` for document processing

---

### 4. Analysis Agent (`analysis_agent.py`)

**Purpose:** Quantitative portfolio analysis

**Key Functions:**
- Calculates portfolio allocation by region/sector
- Computes risk metrics: volatility, Sharpe ratio, VaR (95%, 99%), HHI concentration
- Identifies earnings surprises (beats/misses)
- Analyzes sentiment trends across regions
- Generates morning briefs with recommendations
- **Outputs:** `morning_brief_TIMESTAMP.json`

**Technologies:**
- Direct JSON parsing (no string manipulation)
- `numpy` for statistical calculations
- `pandas` for data aggregation
- Optional RAG integration via Retriever Agent

---

### 5. Language Agent (`language_agent.py`)

**Purpose:** Generate natural language insights using LLM

**Key Functions:**
- Loads data from all agents (API, Scraping, Analysis)
- Constructs comprehensive context for LLM
- Generates morning briefs (3-4 paragraphs)
- Answers specific queries using RAG
- Implements rate limiting (2 req/min for Gemini)
- Response caching (1-hour TTL)
- **Outputs:** `language_report_TIMESTAMP.json`, `llm_cache.json`

**Technologies:**
- `google-generativeai` (Gemini 2.0 Flash)
- Rate limiter with request queue
- UTF-8 encoding for Windows compatibility

---

### 6. Voice Agent (`voice_agent.py`)

**Purpose:** Hands-free portfolio interaction

**Key Functions:**
- Speech-to-text using Google Speech Recognition
- Ambient noise calibration
- Query validation with portfolio keywords
- Text-to-speech using gTTS
- Audio playback with pygame
- Conversation history tracking
- Fallback to text input
- **Outputs:** `voice_conversation_TIMESTAMP.json`

**Technologies:**
- `speech_recognition` for STT
- `gTTS` for TTS
- `pygame` for audio playback
- Integrates with Language Agent for processing

---

### 7. Orchestrator Agent (`orchestrator.py`)

**Purpose:** Coordinate multi-agent workflows

**Key Functions:**
- Checks agent availability
- Executes predefined workflows:
  - **Full Pipeline:** All 6 agents sequentially
  - **Morning Brief:** Quick daily summary
  - **Voice Query:** Interactive voice session
  - **Quick Analysis:** API + Analysis only
  - **Sentiment Check:** Scraping for specific stocks/regions
- Handles file paths across directories
- Logs execution history
- **Outputs:** `orchestrator_log_TIMESTAMP.json`

**Technologies:**
- Python `pathlib` for cross-platform paths
- Dynamic agent loading
- Workflow state management

---

## 🛠️ Technical Stack

### Core Technologies:

- **Python 3.8+**
- **Data Collection:** `yfinance`, `requests`, `feedparser`, `BeautifulSoup`
- **Sentiment Analysis:** Custom keyword-based analyzer
- **Vector Store:** FAISS, `sentence-transformers`
- **LLM:** `google-generativeai` (Gemini 2.0 Flash)
- **Voice:** `speech_recognition`, `gTTS`, `pygame`
- **Web UI:** `streamlit`, `plotly`, `pandas`
- **Scraping:** Firecrawl MCP API (optional)
