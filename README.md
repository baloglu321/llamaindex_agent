🦙 LlamaIndex ReAct Agent (Ollama & Cloudflare)

This repository contains a multi-modal AI Agent built using LlamaIndex and Ollama, featuring a ReAct (Reasoning and Acting) architecture.

The agent is managed by a locally running LLM (such as gemma3:27b or accessed remotely via a Cloudflare Tunnel) and possesses various capabilities including visual analysis, video processing, and file management. It uses Gradio as its user interface.

🚀 Capabilities (Tools)

The agent can perform complex tasks using the following tools:

🧠 Local Intelligence: Uses gemma3:27b (or another preferred model) for logical reasoning.

👁️ Visual Analysis (Vision): Uses the llava model to analyze images and answer questions.

📹 YouTube Integration: Downloads videos, extracts audio, and converts it to text (transcript) using Whisper.

🎬 Video Frame Analysis: Captures frames from specific seconds of videos and explains what is happening.

🔍 Web Search: Accesses up-to-date information via DuckDuckGo and Wikipedia integration.

📂 File Processing: Downloads Excel, JSON, and other files, reads their content, and analyzes them. (Includes local caching feature).

🌤️ Weather: Provides instant weather information via the WeatherStack API.

🧮 Math: Performs basic mathematical operations accurately.

🛠️ Architecture

The project is built on a structure where the LLM runs on a local machine, and the Agent logic (this repo) accesses this LLM via a Cloudflare Tunnel.

graph TD
    User[User] -->|Web UI| Gradio[app.py]
    Gradio -->|Command| Agent[agent.py / ReAct Agent]
    Agent -->|Visual/Text| Tools[Tool Set]
    Tools -->|Download/Search| Internet
    Agent -->|LLM Request| CF[Cloudflare Tunnel URL]
    CF -->|Tunnel| LocalPC[Local Server/PC]
    LocalPC -->|Inference| Ollama[Ollama (Gemma/Llava)]


📋 Requirements

Before running the project, you need to have the following installed/ready on your system:

Ollama Server: The machine where the LLMs are running.

Required Models: ollama pull gemma3:27b and ollama pull llava

Cloudflare Tunnel: A URL exposing your Ollama server port (usually 11434) to the outside (e.g., https://your-tunnel.trycloudflare.com).

System Tools:

ffmpeg (Mandatory for audio and video processing).

Python 3.10+

📦 Installation

Clone the repository:

git clone [https://github.com/baloglu321/llamaindex_agent.git](https://github.com/baloglu321/llamaindex_agent.git)
cd llamaindex_agent


Install dependencies:

pip install -r requirements.txt


(Core packages: llama-index, gradio, nest_asyncio, yt-dlp, openai-whisper, pandas, requests, pillow, opencv-python)

⚙️ Configuration

Open the agent.py file and update the following variables according to your environment:

# agent.py

# Cloudflare Tunnel address of your Ollama server (or http://localhost:11434 if local)
CLOUDFLARE_TUNNEL_URL = "[https://your-created-tunnel-address.trycloudflare.com](https://your-created-tunnel-address.trycloudflare.com)"

# Main Model to be used (Text)
OLLAMA_MODEL_ID = "gemma3:27b"

# Your WeatherStack API Key
WEATHER_API = "your_api_key"


🖥️ Usage

Start the Gradio interface to interact with the agent:

python app.py


This command will provide a local web address in the terminal (usually http://127.0.0.1:7860). Open this address in your browser to chat with the Agent, upload files, or perform video analysis.

📂 Project Structure

agent.py: The brain of the project. Contains the LlamaIndex ReAct agent logic, all tool definitions, and LLM connection settings.

app.py: Gradio-based web interface presenting the Agent to the end-user.

system_prompt.txt: System instructions defining the agent's persona and rules.

requirements.txt: Required Python libraries.

🤝 Contributing

Feel free to send a Pull Request to report bugs or add new features.
