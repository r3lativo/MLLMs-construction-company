### Pseudocode for "MLLMs Construction Company" Project

---

#### **1. Data Preprocessing**

**Goal**: Align dialogues, logs, and screenshots to create a structured dataset.

1. Define the dataset directory path.
2. For each day in the dataset:
   1. Extract `dialogue-with-actions.txt`, `logs.zip`, and `screenshots.zip`.
   2. Parse the dialogue file to group conversations by session ID.
   3. Unzip logs and screenshots into respective folders.
   4. For each session ID:
      1. Load the dialogue text and action sequence.
      2. Parse the log file for environment states and timestamps.
      3. Map dialogue lines and actions to corresponding screenshots using timestamps.
      4. Save the aligned data for each session in a structured format.

---

#### **2. MLLM Agents Framework**

**Goal**: Load the models

1. Initialize the multimodal models with a class.
2. Define functions to call to load prompt, images etc.

---

#### **3. Evaluation Framework**

**Goal**: Analyze interaction quality and task success.

1. Define evaluation metrics:
   1. **Task Success**: Check if the final structure matches the target.
   2. **Dialogue Efficiency**: Measure instruction clarity and brevity.
   3. **Error Recovery**: Assess responses to Builder’s clarification requests.
2. For each session:
   1. Load the interaction log.
   2. Compare final environment state with the target structure.
   3. Analyze dialogue for:
      1. Redundant or ambiguous instructions.
      2. Instances of successful error recovery.
3. Generate a performance report for each session.
4. Aggregate results across all sessions for system-level analysis.

---

#### **4. Main Script**

**Goal**: Orchestrate all components.

1. For each session:
   1. Initialize the Architect and Builder agents.
   2. Simulate the interaction turn by turn:
      1. Architect generates instructions.
      2. Builder executes actions and provides feedback.
      3. Update the environment using the screenshot system.
   3. Log the interaction for evaluation.
2. Call the `evaluation` module to analyze results.
3. Save final reports and insights.