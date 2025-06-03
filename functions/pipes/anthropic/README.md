# 🤖 Anthropic Manifold Pipe

> **Seamless integration between OpenWebUI and Anthropic's Claude models with advanced image processing**

[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/open-webui/functions)
[![OpenWebUI](https://img.shields.io/badge/OpenWebUI-0.3.17+-green.svg)](https://github.com/open-webui/open-webui)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![Anthropic](https://img.shields.io/badge/Anthropic-API-orange.svg)](https://docs.anthropic.com)

---

## 🌟 Overview

**Anthropic Manifold Pipe** is a sophisticated OpenWebUI pipe that enables seamless integration with Anthropic's Claude models through their official API. This pipe provides comprehensive support for text and vision capabilities, intelligent model discovery, efficient caching, and robust error handling.

### ✨ Key Features

- 🔗 **Direct API Integration** - Native connection to Anthropic's Claude models via official API
- 🖼️ **Advanced Image Processing** - Support for both base64 and URL images with size validation
- 📡 **Real-time Streaming** - Live response streaming for immediate feedback
- 🗃️ **Intelligent Caching** - Smart model discovery with configurable cache TTL
- 🛡️ **Robust Error Handling** - Comprehensive exception management and user feedback
- ⚙️ **Parameter Mapping** - Seamless translation between OpenWebUI and Anthropic formats
- 🎯 **System Message Support** - Proper handling of system prompts and instructions
- 📊 **Resource Management** - Image size limits and memory optimization

---

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🏗️ Installation](#️-installation)
- [🎯 Core Concepts](#-core-concepts)
  - [Manifold Architecture](#manifold-architecture)
  - [Model Discovery](#model-discovery)
  - [Image Processing](#image-processing)
- [🛠️ Configuration](#️-configuration)
  - [API Key Setup](#api-key-setup)
  - [Environment Variables](#environment-variables)
  - [Cache Configuration](#cache-configuration)
- [💡 Usage Guide](#-usage-guide)
  - [Basic Text Chat](#basic-text-chat)
  - [Vision Capabilities](#vision-capabilities)
  - [Streaming Responses](#streaming-responses)
- [🏗️ System Architecture](#️-system-architecture)
  - [Request Processing](#request-processing)
  - [Response Handling](#response-handling)
  - [Error Management](#error-management)
- [🔧 Troubleshooting](#-troubleshooting)
- [🚀 Advanced Features](#-advanced-features)
- [🤝 Contributing](#-contributing)

---

## 🚀 Quick Start

### 1️⃣ Install the Pipe
1. Copy the complete pipe code
2. Add as a new pipe in OpenWebUI
3. Configure your Anthropic API key
4. Enable the pipe in your workspace

### 2️⃣ Configure API Access
- Set `ANTHROPIC_API_KEY` in the valves
- Or set `ANTHROPIC_API_KEY` environment variable
- Obtain API key from [Anthropic Console](https://console.anthropic.com)

### 3️⃣ Select Claude Model
- Choose from automatically discovered models
- Examples: `claude-3-sonnet-20240229`, `claude-3-opus-20240229`
- Models appear with `anthropic/` prefix

### 4️⃣ Start Chatting
- Text conversations work immediately
- Upload images for vision capabilities
- Enable streaming for real-time responses

---

## 🏗️ Installation

### Prerequisites
- OpenWebUI version 0.3.17 or higher
- Valid Anthropic API key with appropriate credits
- Python environment with required dependencies
- Administrator access to add pipes

### Step-by-Step Installation

1. **Access Pipe Management**
   - Navigate to OpenWebUI Settings
   - Go to Admin Panel → Pipes
   - Click "Add Pipe"

2. **Install Anthropic Manifold Pipe**
   - Copy the complete pipe code
   - Paste into the pipe editor
   - Set pipe name: "Anthropic Manifold Pipe"
   - Save and enable the pipe

3. **Configure API Authentication**
   - **Method 1**: Set in Valves
     ```
     ANTHROPIC_API_KEY: "your-api-key-here"
     ```
   - **Method 2**: Environment Variable
     ```bash
     export ANTHROPIC_API_KEY="your-api-key-here"
     ```

4. **Verify Installation**
   - Check that Claude models appear in model list
   - Test with a simple text conversation
   - Verify image upload functionality

5. **Optional Configuration**
   - Set model cache TTL via `ANTHROPIC_MODEL_CACHE_TTL`
   - Configure timeout and retry settings
   - Adjust image processing limits

---

## 🎯 Core Concepts

### Manifold Architecture

The **Anthropic Manifold Pipe** implements OpenWebUI's manifold pattern for external provider integration:

#### 🏗️ Component Structure
```python
class Pipe:
    def __init__(self):
        self.type = "manifold"           # Declares as manifold pipe
        self.id = "anthropic"            # Unique identifier
        self.name = "anthropic/"         # Model prefix
    
    def pipes(self) -> List[dict]:       # Model discovery
    def pipe(self, body: dict):          # Request processing
```

#### 🔧 Core Components
- **Model Discovery Engine** - Automatic Claude model detection
- **Request Translator** - OpenWebUI ↔ Anthropic format conversion
- **Image Processor** - Advanced vision capability handling
- **Streaming Manager** - Real-time response delivery
- **Cache System** - Intelligent model list caching
- **Error Handler** - Comprehensive exception management

### Model Discovery

The pipe implements intelligent model discovery with caching:

#### 🔍 Discovery Process
```python
def get_anthropic_models_from_api(self, force_refresh: bool = False) -> List[Dict[str, str]]:
    # 1. Check cache validity
    current_time = time.time()
    if not force_refresh and cache is valid:
        return self._model_cache
    
    # 2. Fetch from Anthropic API
    response = requests.get("https://api.anthropic.com/v1/models")
    
    # 3. Process and cache results
    models = [{"id": model["id"], "name": model["display_name"]} for model in data]
    self._model_cache = models
    
    return models
```

#### 📊 Model Information
| Field | Description | Example |
|-------|-------------|---------|
| `id` | Anthropic model identifier | `claude-3-sonnet-20240229` |
| `name` | Human-readable display name | `Claude 3 Sonnet` |
| `prefix` | OpenWebUI model prefix | `anthropic/claude-3-sonnet` |

### Image Processing

The pipe provides sophisticated image handling with validation:

#### 🖼️ Supported Image Formats
| Format Type | Processing Method | Size Limit | Example |
|-------------|------------------|------------|---------|
| **Base64** | Inline data processing | 5MB per image | `data:image/jpeg;base64,/9j/4AA...` |
| **URL** | Remote image fetching | 5MB per image | `https://example.com/image.jpg` |
| **Multiple Images** | Batch processing | 100MB total | Up to ~20 images |

#### 🔍 Image Validation Logic
```python
def process_image(self, image_data):
    if image_data["image_url"]["url"].startswith("data:image"):
        # Base64 image processing
        mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
        image_size = len(base64_data) * 3 / 4  # Convert to bytes
        
        if image_size > self.MAX_IMAGE_SIZE:  # 5MB limit
            raise ValueError(f"Image size exceeds 5MB limit: {image_size:.2f}MB")
    else:
        # URL image validation
        response = requests.head(url, allow_redirects=True)
        content_length = int(response.headers.get("content-length", 0))
        
        if content_length > self.MAX_IMAGE_SIZE:
            raise ValueError(f"Image at URL exceeds 5MB limit: {content_length:.2f}MB")
```

---

## 🛠️ Configuration

### API Key Setup

#### 🔑 Authentication Methods

**Method 1: Valves Configuration (Recommended)**
```python
# In OpenWebUI Pipe Settings
ANTHROPIC_API_KEY = "sk-ant-api03-xxxxxxxxxxxxxxxxxxxx"
```

**Method 2: Environment Variable**
```bash
# System environment
export ANTHROPIC_API_KEY="sk-ant-api03-xxxxxxxxxxxxxxxxxxxx"

# Docker environment
-e ANTHROPIC_API_KEY="sk-ant-api03-xxxxxxxxxxxxxxxxxxxx"

# Docker Compose
environment:
  - ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxx
```

#### 🛡️ API Key Security
- **Never commit API keys** to version control
- **Use environment variables** in production
- **Rotate keys regularly** for security
- **Monitor usage** in Anthropic Console

### Environment Variables

#### ⚙️ Configuration Options
| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `ANTHROPIC_API_KEY` | `""` | API authentication key | `sk-ant-api03-...` |
| `ANTHROPIC_MODEL_CACHE_TTL` | `600` | Model cache lifetime (seconds) | `300` |

#### 🔧 Advanced Configuration
```bash
# Model caching (10 minutes)
export ANTHROPIC_MODEL_CACHE_TTL=600

# Custom timeout settings (if implemented)
export ANTHROPIC_TIMEOUT=60
export ANTHROPIC_MAX_RETRIES=3
```

### Cache Configuration

#### 🗃️ Model Cache System
```python
# Cache configuration
self._model_cache: Optional[List[Dict[str, str]]] = None
self._model_cache_time: float = 0
self._cache_ttl = int(os.getenv("ANTHROPIC_MODEL_CACHE_TTL", "600"))
```

#### ⚡ Cache Benefits
- **Reduced API Calls** - Minimizes model discovery requests
- **Faster Loading** - Immediate model list availability
- **Rate Limit Protection** - Prevents excessive API usage
- **Offline Resilience** - Works with cached data during outages

---

## 💡 Usage Guide

### Basic Text Chat

#### 💬 Simple Conversation
```json
{
  "model": "anthropic/claude-3-sonnet-20240229",
  "messages": [
    {"role": "user", "content": "Hello, how are you today?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}
```

#### 🎯 System Message Usage
```json
{
  "model": "anthropic/claude-3-sonnet-20240229", 
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to sort a list."}
  ]
}
```

### Vision Capabilities

#### 📸 Image Upload Support
**Base64 Image:**
```json
{
  "model": "anthropic/claude-3-sonnet-20240229",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What's in this image?"},
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
        }
      }
    ]
  }]
}
```

**URL Image:**
```json
{
  "model": "anthropic/claude-3-sonnet-20240229",
  "messages": [{
    "role": "user", 
    "content": [
      {"type": "text", "text": "Analyze this image:"},
      {
        "type": "image_url",
        "image_url": {
          "url": "https://example.com/image.jpg"
        }
      }
    ]
  }]
}
```

### Streaming Responses

#### 🌊 Real-Time Output
```python
# Enable streaming
{
  "model": "anthropic/claude-3-sonnet-20240229",
  "messages": [...],
  "stream": True
}

# Streaming response handling
for chunk in response:
    if chunk["type"] == "content_block_delta":
        text_chunk = chunk["delta"]["text"]
        # Display text_chunk in real-time
```

---

## 🏗️ System Architecture

### Request Processing

#### 🔄 Processing Pipeline
```mermaid
graph TD
    A[OpenWebUI Request] --> B[Extract System Message]
    B --> C[Process Message Content]
    C --> D{Contains Images?}
    D -->|Yes| E[Validate Image Sizes]
    D -->|No| F[Convert to Anthropic Format]
    E --> G[Process Each Image]
    G --> F
    F --> H[Add Parameters]
    H --> I{Streaming Enabled?}
    I -->|Yes| J[Stream Response]
    I -->|No| K[Single Response]
    J --> L[Yield Chunks]
    K --> M[Return Complete Response]
```

### Response Handling

#### 📨 Response Processing
**Streaming Response:**
```python
def stream_response(self, url, headers, payload):
    with requests.post(url, headers=headers, json=payload, stream=True) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if data["type"] == "content_block_delta":
                    yield data["delta"]["text"]
```

**Non-Streaming Response:**
```python
def non_stream_response(self, url, headers, payload):
    response = requests.post(url, headers=headers, json=payload)
    res = response.json()
    return res["content"][0]["text"]
```

### Error Management

#### 🛡️ Exception Handling Strategy
```python
# API Key Validation
if not self.valves.ANTHROPIC_API_KEY:
    return [{"id": "error", "name": "ANTHROPIC_API_KEY is not set..."}]

# HTTP Error Handling
if response.status_code != 200:
    raise Exception(f"HTTP Error {response.status_code}: {response.text}")

# Image Size Validation
if image_size > self.MAX_IMAGE_SIZE:
    raise ValueError(f"Image size exceeds 5MB limit: {image_size:.2f}MB")

# Request Exception Handling
except requests.exceptions.RequestException as e:
    return f"Error: Request failed: {e}"
```

---

## 🔧 Troubleshooting

### Common Issues

#### ❌ API Key Problems
**Problem**: Models not loading or "API Key not set" error
```
Solution: Verify API key configuration
1. Check ANTHROPIC_API_KEY is set in valves or environment
2. Verify API key format: starts with "sk-ant-api03-"
3. Confirm API key is active in Anthropic Console
4. Check API key has sufficient credits
5. Test API key with curl command
```

#### ❌ Image Upload Failures
**Problem**: Images not processing or size errors
```
Solution: Check image specifications
1. Verify image size is under 5MB per image
2. Check total images under 100MB combined
3. Ensure image format is supported (JPEG, PNG, GIF, WebP)
4. For URL images, verify URL is accessible
5. Check base64 encoding is valid
```

#### ❌ Model Discovery Issues
**Problem**: No Claude models appearing in list
```
Solution: Debug model discovery
1. Check API key permissions for model access
2. Verify network connectivity to api.anthropic.com
3. Check cache TTL and force refresh
4. Review console logs for API errors
5. Test direct API call to /v1/models endpoint
```

#### ❌ Streaming Problems
**Problem**: Streaming responses not working
```
Solution: Verify streaming configuration
1. Ensure stream=True in request body
2. Check network supports streaming connections
3. Verify timeout settings are appropriate
4. Test with non-streaming first
5. Check firewall/proxy streaming support
```

### Debug Mode

#### 🐛 Enhanced Debugging
```python
def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
    print(f"=== ANTHROPIC PIPE DEBUG ===")
    print(f"Model: {body['model']}")
    print(f"Message count: {len(body['messages'])}")
    print(f"Stream enabled: {body.get('stream', False)}")
    
    # Image processing debug
    for i, message in enumerate(body["messages"]):
        if isinstance(message.get("content"), list):
            image_count = sum(1 for item in message["content"] if item["type"] == "image_url")
            print(f"Message {i}: {image_count} images")
    
    # API call debug
    print(f"API URL: {url}")
    print(f"Payload keys: {list(payload.keys())}")
    
    return response
```

#### 📊 Debug Information
- **Request Details** - Model, parameters, message structure
- **Image Processing** - Size validation, format detection
- **API Communication** - Headers, payload, response status
- **Caching Status** - Cache hits, misses, refresh events
- **Error Context** - Full exception details and stack traces

---

## 🚀 Advanced Features

### Custom Model Configuration

#### 🎯 Extended Model Support
```python
def get_anthropic_models_from_api(self, force_refresh: bool = False) -> List[Dict[str, str]]:
    # Add custom model filtering
    models = []
    for model in data.get("data", []):
        # Filter for specific model types
        if self.is_supported_model(model["id"]):
            models.append({
                "id": model["id"],
                "name": self.get_display_name(model),
                "capabilities": self.get_model_capabilities(model)
            })
    
    return models

def is_supported_model(self, model_id: str) -> bool:
    # Define supported model patterns
    supported_patterns = [
        "claude-3-opus",
        "claude-3-sonnet", 
        "claude-3-haiku"
    ]
    return any(pattern in model_id for pattern in supported_patterns)
```

### Advanced Image Processing

#### 🖼️ Enhanced Image Handling
```python
def process_image_batch(self, images: List[dict]) -> List[dict]:
    """Process multiple images with optimization."""
    processed_images = []
    total_size = 0
    
    for image in images:
        processed = self.process_image(image)
        processed_images.append(processed)
        
        # Size tracking
        if processed["source"]["type"] == "base64":
            image_size = len(processed["source"]["data"]) * 3 / 4
            total_size += image_size
            
            if total_size > self.MAX_TOTAL_IMAGE_SIZE:
                raise ValueError("Total image size exceeds 100MB limit")
    
    return processed_images

def optimize_image_quality(self, image_data: str, target_size: int) -> str:
    """Optimize image quality to meet size constraints."""
    # Implement image compression if needed
    pass
```

### Performance Optimization

#### ⚡ Enhanced Caching
```python
class ModelCacheManager:
    def __init__(self):
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
    
    def get_models(self, api_key_hash: str) -> Optional[List[Dict]]:
        cache_key = f"models_{api_key_hash}"
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry["timestamp"] < self.ttl:
                self.cache_stats["hits"] += 1
                return entry["data"]
        
        self.cache_stats["misses"] += 1
        return None
    
    def set_models(self, api_key_hash: str, models: List[Dict]):
        cache_key = f"models_{api_key_hash}"
        self.cache[cache_key] = {
            "data": models,
            "timestamp": time.time()
        }
```

#### 🔄 Connection Pooling
```python
import requests.adapters
from urllib3.util.retry import Retry

class AnthropicAPIClient:
    def __init__(self):
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=retry_strategy
        )
        
        self.session.mount("https://", adapter)
```

### Integration Patterns

#### 🔗 Multi-Model Routing
```python
def intelligent_model_selection(self, request: dict) -> str:
    """Select optimal Claude model based on request characteristics."""
    content = self.extract_content(request["messages"])
    
    # Analyze request complexity
    if self.has_images(request):
        return "claude-3-sonnet-20240229"  # Best vision model
    elif len(content) > 10000:
        return "claude-3-opus-20240229"    # Best for complex tasks
    elif self.is_simple_query(content):
        return "claude-3-haiku-20240229"   # Fast for simple tasks
    else:
        return "claude-3-sonnet-20240229"  # Balanced default
```

#### 📊 Usage Analytics
```python
def track_usage(self, model: str, tokens_used: int, response_time: float):
    """Track API usage for optimization."""
    usage_data = {
        "model": model,
        "tokens": tokens_used,
        "response_time": response_time,
        "timestamp": time.time()
    }
    
    # Log to analytics system
    self.analytics_logger.log(usage_data)
    
    # Update usage statistics
    self.update_model_stats(model, tokens_used, response_time)
```

---

## 🤝 Contributing

### Development Setup

#### 🛠️ Local Development
1. **Clone Repository** - Set up local OpenWebUI development environment
2. **Install Dependencies** - Ensure requests, pydantic are available
3. **API Access** - Obtain Anthropic API key for testing
4. **Test Environment** - Use OpenWebUI development instance
5. **Submit PR** - Follow OpenWebUI contribution guidelines

### Pipe Guidelines

#### 📝 Best Practices
- **Efficient API Usage** - Minimize unnecessary requests
- **Robust Error Handling** - Graceful failure management
- **Security Focus** - Secure API key handling
- **Performance Optimization** - Caching and connection pooling
- **Comprehensive Testing** - Multi-model and multi-format testing

#### 🧪 Testing Requirements
- **API Integration** - Test with real Anthropic API
- **Image Processing** - Verify all image formats and sizes
- **Streaming Support** - Confirm real-time response delivery
- **Error Scenarios** - Test invalid keys, network failures
- **Model Discovery** - Verify dynamic model loading

### Feature Requests

#### 💡 Enhancement Ideas
- **Function Calling** - Support for Anthropic function calling
- **Custom Parameters** - Extended parameter support
- **Advanced Caching** - Persistent cache storage
- **Load Balancing** - Multiple API key rotation
- **Usage Monitoring** - Detailed analytics and reporting

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **@justinh-rahb** - Lead author and development
- **@christian-taillon** - Co-author and contributions
- **@jfbloom22** - Co-author and enhancements
- **OpenWebUI Team** - For the amazing pipe system architecture
- **Anthropic** - For providing the Claude API and excellent documentation
- **Community Contributors** - For feedback and testing

---

## 📞 Support

- **GitHub Issues** - [Report bugs and request features](https://github.com/open-webui/functions/issues)
- **Discussions** - [Community support and questions](https://github.com/open-webui/functions/discussions)
- **Documentation** - [OpenWebUI Pipes Documentation](https://docs.openwebui.com)
- **Anthropic Docs** - [Anthropic API Documentation](https://docs.anthropic.com)

---

<div align="center">

**🤖 Enhance your AI conversations with Claude's intelligence!**

*Seamless integration • Advanced vision • Real-time streaming*

</div>