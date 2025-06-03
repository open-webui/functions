# OpenWebUI + LangGraph Research Agent: Simple Self-Hosted Solution

This guide shows you how to convert the Gemini LangGraph quickstart to use **OpenWebUI directly** as the interface, with local Ollama models and SearXNG search. This approach is much simpler than maintaining the original React frontend and provides a better user experience.

## Why OpenWebUI is Perfect for This

**OpenWebUI already has everything we need:**
- Native LangGraph support through Pipelines framework
- Built-in SearXNG search integration
- Excellent local model support with Ollama
- ChatGPT-like interface that users love
- Simple deployment with Docker

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐    ┌─────────────┐
│  OpenWebUI  │───▶│  Pipeline   │───▶│  LangGraph  │
│  Interface  │     │  (Bridge)   │    │   Agent     │
└─────────────┘     └─────────────┘    └─────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Ollama    │    │   SearXNG   │    │ PostgreSQL  │
│   Models    │    │   Search    │    │   State     │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Complete Implementation

### Step 1: Docker Compose Setup

Create a `docker-compose.yml` that runs everything:

```yaml
version: '3.8'
services:
  # Core OpenWebUI with Ollama
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - ENABLE_RAG_WEB_SEARCH=true
      - RAG_WEB_SEARCH_ENGINE=searxng
      - SEARXNG_QUERY_URL=http://searxng:8080/search?q=<query>&format=json
      - WEBUI_SECRET_KEY=your-secret-key-here
      - PIPELINES_URLS=http://pipelines:9099
    volumes:
      - open-webui:/app/backend/data
    depends_on:
      - ollama
      - searxng
      - pipelines
    restart: unless-stopped

  # Ollama for local models
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  # SearXNG for web search
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8080:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://localhost:8080/
    depends_on:
      - searxng-redis
    restart: unless-stopped

  searxng-redis:
    image: redis:alpine
    container_name: searxng-redis
    command: redis-server --save 30 1 --loglevel warning
    volumes:
      - searxng-redis:/data
    restart: unless-stopped

  # OpenWebUI Pipelines for LangGraph
  pipelines:
    image: ghcr.io/open-webui/pipelines:main
    container_name: pipelines
    ports:
      - "9099:9099"
    environment:
      - PIPELINES_URLS=https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/langgraph_research_pipeline.py
    volumes:
      - pipelines:/app/pipelines
    depends_on:
      - postgres
    restart: unless-stopped

  # PostgreSQL for LangGraph state persistence
  postgres:
    image: postgres:16-alpine
    container_name: postgres
    environment:
      - POSTGRES_DB=langgraph
      - POSTGRES_USER=langgraph
      - POSTGRES_PASSWORD=langgraph
    volumes:
      - postgres:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  open-webui:
  ollama:
  searxng-redis:
  pipelines:
  postgres:
```

### Step 2: SearXNG Configuration

Configure SearXNG to enable JSON format for OpenWebUI integration:

Create `searxng/settings.yml`:
```yaml
use_default_settings: true
search:
  formats:
    - html
    - json  # Required for OpenWebUI integration
server:
  secret_key: "your-secret-key-here"
  limiter: true
redis:
  url: "redis://searxng-redis:6379/0"
engines:
  - name: google
    weight: 1
  - name: bing
    weight: 1
  - name: duckduckgo
    weight: 1
```

### Step 3: LangGraph Research Pipeline

Create a custom OpenWebUI pipeline that integrates the LangGraph research agent:

Create `langgraph_research_pipeline.py`:

```python
"""
title: LangGraph Research Agent Pipeline
author: Your Name
date: 2025-01-03
version: 1.0
license: MIT
description: A research agent pipeline using LangGraph with local models and SearXNG
requirements: langgraph, langchain-ollama, psycopg2, requests, beautifulsoup4
"""

from typing import List, Union, Generator, Iterator, Dict, Any, Annotated
import operator
import requests
import json
import logging
from urllib.parse import urljoin

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        # Model configuration
        ollama_base_url: str = Field(
            default="http://ollama:11434",
            description="Ollama base URL"
        )
        model_name: str = Field(
            default="llama3.1:8b",
            description="Local model to use"
        )
        
        # Search configuration
        searxng_base_url: str = Field(
            default="http://searxng:8080",
            description="SearXNG base URL"
        )
        max_search_results: int = Field(
            default=5,
            description="Maximum search results per query"
        )
        max_research_iterations: int = Field(
            default=3,
            description="Maximum research iterations"
        )
        
        # Database configuration
        postgres_url: str = Field(
            default="postgresql://langgraph:langgraph@postgres:5432/langgraph",
            description="PostgreSQL connection URL for state persistence"
        )

    def __init__(self):
        self.id = "langgraph_research"
        self.name = "LangGraph Research Agent"
        self.valves = self.Valves()
        
        # Initialize components
        self.llm = None
        self.graph = None
        self.checkpointer = None
        
        # Initialize on startup
        self._initialize_components()

    def _initialize_components(self):
        """Initialize LLM, checkpointer, and graph"""
        try:
            # Initialize LLM
            self.llm = ChatOllama(
                model=self.valves.model_name,
                base_url=self.valves.ollama_base_url,
                temperature=0.7
            )
            
            # Initialize PostgreSQL checkpointer for state persistence
            self.checkpointer = PostgresSaver.from_conn_string(
                self.valves.postgres_url
            )
            
            # Build the research graph
            self.graph = self._build_research_graph()
            
            logger.info("LangGraph Research Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

    def _build_research_graph(self):
        """Build the LangGraph research workflow"""
        
        # Define the state structure
        class ResearchState(BaseModel):
            messages: Annotated[List[BaseMessage], operator.add]
            query: str = ""
            search_queries: List[str] = []
            search_results: List[Dict] = []
            research_complete: bool = False
            iteration_count: int = 0
            
        # Node functions
        def query_generation_node(state: ResearchState):
            """Generate search queries from user input"""
            prompt = f"""Generate 2-3 specific, diverse search queries to thoroughly research: {state.query}
            
            Return only the search queries, one per line.
            Make them specific and complementary to cover different aspects."""
            
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                queries = [q.strip() for q in response.content.split('\n') if q.strip()]
                return {"search_queries": queries}
            except Exception as e:
                logger.error(f"Query generation error: {e}")
                return {"search_queries": [state.query]}

        def web_research_node(state: ResearchState):
            """Perform web research using SearXNG"""
            results = []
            
            for query in state.search_queries:
                try:
                    # Search using SearXNG
                    search_url = f"{self.valves.searxng_base_url}/search"
                    params = {
                        'q': query,
                        'format': 'json',
                        'categories': 'general',
                        'engines': 'google,bing,duckduckgo'
                    }
                    
                    response = requests.get(search_url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        for result in data.get('results', [])[:self.valves.max_search_results]:
                            results.append({
                                'title': result.get('title', ''),
                                'url': result.get('url', ''),
                                'content': result.get('content', '')[:500],  # Truncate
                                'query': query
                            })
                    
                except Exception as e:
                    logger.error(f"Search error for query '{query}': {e}")
                    continue
            
            return {"search_results": results}

        def analysis_node(state: ResearchState):
            """Analyze search results and determine if more research is needed"""
            search_summary = "\n".join([
                f"Query: {r['query']}\nTitle: {r['title']}\nContent: {r['content']}\n"
                for r in state.search_results
            ])
            
            prompt = f"""Original question: {state.query}
            
            Research results so far:
            {search_summary}
            
            Iteration: {state.iteration_count + 1}/{self.valves.max_research_iterations}
            
            Analysis:
            1. Do these results provide sufficient information to answer the original question comprehensively?
            2. Are there important knowledge gaps that need additional research?
            
            Respond with either:
            - "COMPLETE: [brief reason]" if sufficient information is available
            - "CONTINUE: [specific gaps to research]" if more research is needed
            """
            
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                analysis = response.content.strip()
                
                # Check iteration limit
                if state.iteration_count >= self.valves.max_research_iterations - 1:
                    return {"research_complete": True, "iteration_count": state.iteration_count + 1}
                
                # Check if research is complete
                research_complete = analysis.upper().startswith("COMPLETE")
                return {
                    "research_complete": research_complete,
                    "iteration_count": state.iteration_count + 1
                }
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                return {"research_complete": True, "iteration_count": state.iteration_count + 1}

        def synthesis_node(state: ResearchState):
            """Generate comprehensive answer with citations"""
            # Organize results by source
            sources = {}
            for i, result in enumerate(state.search_results, 1):
                sources[i] = {
                    'title': result['title'],
                    'url': result['url'],
                    'content': result['content']
                }
            
            sources_text = "\n".join([
                f"[{i}] {info['title']}\n{info['content']}\nURL: {info['url']}\n"
                for i, info in sources.items()
            ])
            
            prompt = f"""Question: {state.query}
            
            Research Sources:
            {sources_text}
            
            Provide a comprehensive, well-structured answer using the research above.
            
            Requirements:
            - Use citations in the format [1], [2], etc.
            - Provide multiple perspectives when relevant
            - Include a "References" section at the end
            - Be thorough but concise
            - Highlight key findings and insights
            """
            
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                
                # Add references section
                references = "\n\nReferences:\n" + "\n".join([
                    f"[{i}] {info['title']} - {info['url']}"
                    for i, info in sources.items()
                ])
                
                final_answer = response.content + references
                return {"messages": [AIMessage(content=final_answer)]}
                
            except Exception as e:
                logger.error(f"Synthesis error: {e}")
                return {"messages": [AIMessage(content=f"Error generating response: {e}")]}

        # Build the graph
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("generate_queries", query_generation_node)
        workflow.add_node("web_research", web_research_node)
        workflow.add_node("analyze", analysis_node)
        workflow.add_node("synthesize", synthesis_node)
        
        # Add edges
        workflow.add_edge(START, "generate_queries")
        workflow.add_edge("generate_queries", "web_research")
        workflow.add_edge("web_research", "analyze")
        
        # Conditional edge: continue research or synthesize
        def should_continue(state: ResearchState):
            if state.research_complete:
                return "synthesize"
            else:
                return "generate_queries"  # Continue research loop
        
        workflow.add_conditional_edges("analyze", should_continue)
        workflow.add_edge("synthesize", END)
        
        # Compile with checkpointer for state persistence
        return workflow.compile(checkpointer=self.checkpointer)

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline function called by OpenWebUI"""
        
        try:
            # Create a unique thread for this conversation
            thread_config = {
                "configurable": {
                    "thread_id": f"research_{hash(str(messages[-10:]))}"  # Use recent message history as thread ID
                }
            }
            
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=user_message)],
                "query": user_message,
                "search_queries": [],
                "search_results": [],
                "research_complete": False,
                "iteration_count": 0
            }
            
            # Run the research workflow
            final_state = self.graph.invoke(initial_state, config=thread_config)
            
            # Return the final response
            if final_state.get("messages"):
                return final_state["messages"][-1].content
            else:
                return "I apologize, but I encountered an error while researching your question. Please try rephrasing your query."
                
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            return f"Error: {str(e)}"

    async def on_valves_updated(self):
        """Reinitialize when valves are updated"""
        self._initialize_components()
```

### Step 4: Setup and Deployment

**1. Initialize the environment:**
```bash
# Clone and setup
git clone https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart
cd gemini-fullstack-langgraph-quickstart

# Replace the docker-compose.yml with our OpenWebUI version
# Add the langgraph_research_pipeline.py
# Create searxng configuration
mkdir searxng
```

**2. Pull and setup Ollama models:**
```bash
# Start Ollama first
docker-compose up -d ollama

# Wait for it to be ready, then pull models
sleep 30
docker exec ollama ollama pull llama3.1:8b
docker exec ollama ollama pull llama3.2:1b  # Backup model
```

**3. Start all services:**
```bash
# Start everything
docker-compose up -d

# Check that all services are running
docker-compose ps
```

**4. Configure OpenWebUI:**
- Open http://localhost:3000
- Go to Admin Settings → Pipelines
- The LangGraph Research Pipeline should appear automatically
- Go to Admin Settings → Web Search
- Verify SearXNG is configured and working

### Step 5: Usage

**The research agent is now available as a model in OpenWebUI:**

1. **Start a new chat**
2. **Select "LangGraph Research Agent" as your model**
3. **Ask research questions like:**
   - "What are the latest developments in quantum computing?"
   - "Compare different approaches to climate change mitigation"
   - "Analyze the current state of renewable energy adoption globally"

**The agent will:**
- Generate multiple search queries
- Research using SearXNG
- Analyze results for completeness
- Iterate if needed (up to 3 times)
- Provide a comprehensive answer with citations

## Advanced Features

### Multi-Model Support

Add different research agents for different purposes:

```python
# In your pipeline, add multiple model variants
class Pipeline:
    class Valves(BaseModel):
        research_model: str = Field(default="llama3.1:8b", description="Main research model")
        analysis_model: str = Field(default="qwen2.5:14b", description="Analysis model") 
        synthesis_model: str = Field(default="llama3.1:8b", description="Synthesis model")
```

### Custom Search Domains

Configure SearXNG for domain-specific research:

```yaml
# In searxng/settings.yml
engines:
  - name: google_scholar
    engine: google_scholar
    weight: 2
  - name: arxiv
    engine: arxiv
    weight: 2
  - name: wikipedia
    engine: wikipedia 
    weight: 1
```

### Real-time Status Updates

The pipeline includes status updates that show research progress in OpenWebUI:

```python
# Add to your pipeline nodes
await __event_emitter__({
    "type": "status",
    "data": {
        "description": f"🔍 Searching for: {query}",
        "done": False
    }
})
```

## Benefits of This Approach

**1. Simplicity:** Single interface for everything - no separate frontend to maintain

**2. Built-in Features:** OpenWebUI includes user management, chat history, model switching, and more

**3. Extensibility:** Easy to add more pipelines, tools, and models

**4. Self-Hosted:** Complete privacy and control - no external APIs required

**5. Production Ready:** OpenWebUI is battle-tested and actively maintained

## Troubleshooting

**SearXNG 403 Errors:**
Ensure JSON format is enabled in settings.yml

**Pipeline Not Loading:**
Check the logs: `docker-compose logs pipelines`

**Model Issues:**
Verify Ollama models: `docker exec ollama ollama list`

**Database Connection:**
Check PostgreSQL: `docker-compose logs postgres`

This approach gives you the full power of the original Gemini LangGraph research agent, but with a much simpler deployment and a better user experience through OpenWebUI's excellent interface!
