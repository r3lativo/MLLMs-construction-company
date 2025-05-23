# MLLMs Construction Company 👷  
**Investigating Multimodal LLMs' Communicative Skills in a Collaborative Building Task**


⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀     
⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⠾⢻⣿⡟⠻⠶⢦⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀     
⠀⠀⠀⠀⣀⣤⠾⠛⠉⠀⠀⣸⠛⣷⠀⠀⠀⠀⠉⠙⠻⠶⣦⣤⣀⠀⠀⠀⠀⠀     
⠀⠀⠐⠛⠋⠀⠀⠀⠀⠀⠀⠛⠀⠛⠂⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠛⠒⠂⠀⠀     
⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀     
⠀⠀⠀⢠⣤⣤⣤⠀⠀⠀⠀⢠⣤⡄⢠⣤⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⡄⠀⠀⠀     
⠀⠀⠀⠈⠉⠉⠉⠀⠀⠀⠀⠸⠿⠇⠸⠿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀     
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣶⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀     
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀     
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⠀⠀⠀     
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠃⡀⠀⠀     
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠋⠈⠛⠀⠀     
⠀ ⠀⠀⠀⠀⠀⠀⠀⠀⣀⢸⣿⡇⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⡇⠀     
⠀ ⠀⠀⠀⠀⠀⢀⣴⠾⠋⢸⣿⡇⠈⠳⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀     
⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠈⠛⠃⠀⠀⠀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀     



This project features a multi-agent system where an **Architect** plans and instructs, and a **Builder** executes commands to construct structures in a simulated 3D block world.


## 🌟 Features

* **Collaborative AI Agents:** Architect and Builder agents work together, communicating via LLMs.
* **Simulated 3D World:** A simple block-based environment for dynamic building.
* **Intelligent Instruction Following:** Builder interprets natural language instructions and executes structured actions.
* **Resource Management:** Builder tracks and manages its block inventory, dynamically provided with instructions.
* **Multimodal Input (Architect):** Architect can interpret visual blueprints (images) alongside textual structure descriptions.
* **LLM Integration:** Designed for seamless interaction with Large Language Models via the vLLM API.
* **Detailed Logging:** Comprehensive logs for agent actions and world state.


## 🛠️ Setup

### Prerequisites

1.  **Python 3.9+**
2.  **vLLM API Server:** You'll need a [vLLM](https://github.com/vllm-project/vllm) server running with your chosen LLM (e.g., Mistral-Small-3.1-24B-Instruct-2503). Consult vLLM's documentation for setup. Ensure it's accessible at `http://localhost:8000/v1` or update `config/config.yaml`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone -b restructure <https://github.com/r3lativo/MLLMs-construction-company/>
    cd MLLMs-construction-company
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

Serve the model(s) you want to use via vllm

```bash
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
--tokenizer_mode mistral \
--config_format mistral \
--load_format mistral \
--tool-call-parser mistral \
--enable-auto-tool-choice \
--limit_mm_per_prompt 'image=100' \
--tensor-parallel-size 2 \
--dtype bfloat16
```

Run the simulation

```bash
python simulation.py
```


## 💡 How it Works

The simulation involves a continuous feedback loop:

1.  The Architect sends initial building instructions and a blueprint (text and/or images) to the Builder.
2.  The Builder receives instructions, crucially combined with its **current block inventory**.
3.  Based on instructions and resources, the Builder's LLM generates and executes structured actions (e.g., `place_block`, `remove_block`) within the `World`.
4.  The `World` modifies its state based on Builder's actions.
5.  The Builder sends a textual update back to the Architect.
6.  The Architect reviews the Builder's report and receives the **current world state** (JSON and/or rendered images of the world). It then issues further instructions, continuing the cycle until the structure is complete.

