# Langfuse Filter Function v3

**Author:** YetheSamartaka  
**Version:** 1.0.0  
**License:** MIT  
**Date:** 2025-10-27  

---

## Overview

A filter plugin for **Open WebUI (v0.6.32+)** that integrates with **Langfuse v3** for telemetry, tracing, and analytics.  
It logs chat sessions, user inputs, model responses, and token usage directly to Langfuse (Cloud or local).

---

## Features

- Automatic trace creation per chat session  
- Logs user input and assistant responses  
- Tracks token usage (input/output)  
- Supports Langfuse Cloud or local instance  
- Optional debug mode with console logs  
- Custom tags and metadata injection  

## How It Works

### `inlet()`
- Called **before** LLM execution  
- Creates or updates a Langfuse trace  
- Logs user input and metadata  

### `outlet()`
- Called **after** LLM execution  
- Logs assistant response and token usage  
- Finalizes and flushes the trace  

---

## Integration (Open WebUI)

1. Athis file into `filters/langfuse_filter_v3.py` in your Open WebUI instance.  
2. In **Admin → Functions**, add a new function and select either put it there manually or Import it From Link 
3. Set your Langfuse keys and host in the **Valves** settings.  
4. Save and enable it and then either set it as Global or for specific models

All chat activity will then be automatically logged in **Langfuse**.

---

## License

**MIT License**
