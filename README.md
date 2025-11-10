# mem-v-gen

## Installation

1. Install dependencies from the requirements file:

```bash
pip install -r requirement.txt
```

2. Create a `.env` file in the project root with your Neuronpedia API key:

```bash
NEURONPEDIA_API_KEY=your_api_key_here
```

## Usage

Run the program from the main script:

```bash
python main.py
```

## Task Selection

The tool supports two different analysis tasks. Configure the task by setting the `SELECTED_TASK` variable in `main.py`:

### 1. PROMPT_SUBGRAPH_COMPARE

Finds unique and/or overlapping features from the subgraphs of multiple different prompts.

**Configuration in main.py:**
```python
SELECTED_TASK = Task.PROMPT_SUBGRAPH_COMPARE

# Main prompt to analyze
MAIN_PROMPT = "YOUR MEMORIZED PROMPT HERE"

# Prompts where features are expected to NOT be present
DIFF_PROMPTS = [
    "different prompt 1",
    "different prompt 2"
]

# Prompts where features are expected to be present
SIM_PROMPTS = [
    "similar prompt 1",
    "similar prompt 2"
]
```

This task compares the subgraph of the main prompt against:
- **DIFF_PROMPTS**: Finds features unique to the main prompt (not present in these)
- **SIM_PROMPTS**: Finds features common across all similar prompts

### 2. TOKEN_SUBGRAPH_COMPARE

Finds unique features from a token of interest's subgraph compared to subgraphs of other top output tokens.

**Configuration in main.py:**
```python
SELECTED_TASK = Task.TOKEN_SUBGRAPH_COMPARE

# Main prompt to analyze
MAIN_PROMPT = "YOUR MEMORIZED PROMPT HERE"

# Token to focus analysis on
TOKEN_OF_INTEREST = "TOKEN OF INTEREST"

# Number of top output tokens to compare against
TOP_K = N
```

This task analyzes the subgraph of a specific token of interest and compares it against the subgraphs of the top K alternative output tokens to identify distinguishing features.

## Additional Configuration

**Model Settings:**
```python
MODEL = "gemma-2-2b"      # Model name
SUBMODEL = "clt-hp"        # Submodel variant
```

**Data Path:**
```python
SAVE_PATH = "~/data/spar-memory/neuronpedia/"  # Path to graph data
```

## Output

The program will print the Neuronpedia URL to the main prompt's attribution graph. From this URL, you can load and view the identified features as a subgraph.

**Subgraph Names:**

- **TOKEN_SUBGRAPH_COMPARE**: Creates a subgraph named `"Unique Features for [Token of Interest]"`
  - Example: `"Unique Features for MERCHANTABILITY"`

- **PROMPT_SUBGRAPH_COMPARE**: Creates a subgraph named `"Features of Interest (MM-DD-YY HH:MM:SS)"`
  - Example: `"Features of Interest (11-10-25 14:30:45)"`
