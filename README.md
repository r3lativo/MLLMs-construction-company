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



In this project, we implement a collaborative building task where two multimodal large language models (MLLMs) assume the roles of Architect and Builder. The goal is to study how different prompting techniques (text-only, images-only, and mixed modalities) affect the models’ ability to communicate in a human-like manner during a shared building task.

## Overview

The project aims to answer questions such as:
- How effective are MLLMs at replicating human-like communication in a collaborative task?
- Can multimodal prompts improve dialogue quality and task success?
- How do different experimental setups (zero-shot vs. one-shot) compare in performance?

Our experiments are inspired by previous work on collaborative building tasks in Minecraft-like environments and expand the setting to a fully automated scenario where both agents are MLLMs. Detailed methods, experimental design, and results are discussed in the accompanying report.

## Repository Structure

```plaintext
.
├── analysis
│   ├── judge_analysis_BASE.json
│   ├── parsed_actions.json
│   ├── parsed_actions_with_metrics.json
├── data
│   ├── judge_data
│   ├── llava_prompts
│   ├── one_shot_example
│   ├── minecraft_corpus
│   └── structures
├── README.md
├── requirements.txt
├── results
│   └── [multiple log and JSON files from experiments]
└── src
    ├── actions_evaluation.py
    ├── data_analysis.ipynb
    ├── data_preprocessing.py
    ├── judge.py
    ├── judge_utils.py
    ├── main.py
    ├── parse_and_convert.py
    ├── render.py
    ├── render_utils.py
    ├── run_experiments.sh
    ├── utils.py
    └── worldstate_decompile.py
```

- **analysis/**: Contains scripts and output files for evaluating model performance, including parsed actions and metrics.
- **data/**: Houses all necessary input data such as judge prompts, the Minecraft dialogue corpus, and configuration files for target structures.
- **results/**: Stores experiment outputs (JSON logs, evaluation scores, etc.) for different experimental conditions.
- **src/**: Contains the main codebase:
  - `main.py` is the entry point that orchestrates the experiments.
  - `data_preprocessing.py` and `parse_and_convert.py` handle data and log processing.
  - `judge.py` and `judge_utils.py` implement evaluation of dialogue human-likeness.
  - Other modules (e.g., `render.py`, `actions_evaluation.py`) support visualization and action evaluation.
  - `data_analysis.ipynb` provides an interactive environment for further analysis.

## Installation and Setup

**Clone the Repository**

   ```bash
   git clone https://github.com/r3lativo/MLLMs-construction-company.git
   cd MLLMs-construction-company
   ```

Ensure you have Python 3.10+ installed. Then install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage


  Use the provided shell script to run experiments:
  
  ```bash
  bash src/run_experiments.sh
  ```

  Alternatively, you can execute the main Python script, for example with:
  
  ```bash
  python src/main.py --structure_id C1
  ```


  Open the Jupyter notebook `src/data_analysis.ipynb` to explore and visualize experiment results.

## Connection to the Report

This repository underpins the experimental setup detailed in our report:

**Experimental Design**  
  The code implements a collaborative building task where two LLaVA-based MLLMs interact under different input modalities (text-only, images-only, mixed) and learning conditions (zero-shot, one-shot).

**Evaluation Metrics**  
  The evaluation scripts in the `analysis` folder calculate metrics such as task success rate, accuracy, precision, and human-likeness scores.

**Results**  
  Experiment logs and JSON results stored in the `results` directory are analyzed both quantitatively and qualitatively.


## One Shot Example

`<Builder>` Mission has started.<br>
`<Architect>` hello<br>
`<Builder>` hello<br>
`<Architect>` are u rdy to get to work?<br>
`<Builder>` yes<br>
`<Architect>` ok<br>
`<Architect>` build a 2x1 structure that is blue<br>
`<Builder>` is the structure extending upwards?<br>
`<Architect>` no, it goes across<br>
![image 1](data/one_shot_example/images/image_1.png)<br>
![image 2](data/one_shot_example/images/image_2.png)<br>
`<Builder>` is that good?<br>
`<Architect>` now place 1 blue piece on the left block extending upwards<br>
`<Architect>` yes that is correct<br>
![final image](data/one_shot_example/images/final_image.png)<br>
`<Builder>` like that?<br>
`<Architect>` yes, now it is finished<br>
`<Builder>` good job!<br>
`<Architect>` you too builder<br>


## Tables and Results
### Table 1: The six experimental conditions. TS is short for target structure.  


|                | Zero-shot       | One-shot        |
|--------------|---------------|---------------|
| **Text-only**  | TS: JSON      | TS: JSON      |
| **Mixed**      | TS: JSON + Image | TS: JSON + Image |
| **Images-only** | TS: Image      | TS: Image      |


### Table 2: Builders' and architects' most typical communication patterns as recorded in Narayan-Chen (2019).  


| **Builder**                  | **Architect**                  |
|------------------------------|--------------------------------|
| Clarification questions      | References to common shapes   |
|                              | Implicit references           |


### Table 3: Summary of mean human likeness (HL, 1–5) and mean structure matching (accuracy and precision) for one-shot and zero-shot experiments under different IMG and JSON input conditions.  


|                | IMG  | JSON  | HL   | Accuracy | Precision |
|--------------|------|------|------|----------|----------|
| **One-shot**  | No   | JSON  | 1.00 | 0.00     | 0.00     |
|              | IMG  | No    | 1.77 | 0.00     | 0.00     |
|              | IMG  | JSON  | 1.15 | 0.09     | 0.09     |
| **Zero-shot** | No   | JSON  | 1.31 | 0.31     | 0.65     |
|              | IMG  | No    | 1.50 | 0.00     | 0.00     |
|              | IMG  | JSON  | 1.67 | 0.26     | 0.42     |

