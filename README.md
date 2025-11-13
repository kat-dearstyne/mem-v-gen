# mem-v-gen

## Installation

1. Install dependencies from the requirements file:

```bash
pip install -r requirement.txt
```

2. Create a `.env` file in the project root with your Neuronpedia API key and config name:

```bash
NEURONPEDIA_API_KEY=your_api_key_here
CONFIG_NAME=merchantability
```

The `CONFIG_NAME` variable specifies which JSON config file to load from the `configs/` directory (e.g., `CONFIG_NAME=merchantability` loads `configs/merchantability.json`).

## Configuration Files

Prompt configurations are stored as JSON files in the `configs/` directory. Each config file can contain:
- `MAIN_PROMPT`: The main memorized prompt to analyze (required)
- `DIFF_PROMPTS`: Array of contrasting prompts (for PROMPT_SUBGRAPH_COMPARE)
- `SIM_PROMPTS`: Array of similar prompts (for PROMPT_SUBGRAPH_COMPARE)
- `TOKEN_OF_INTEREST`: Specific token to analyze (for TOKEN_SUBGRAPH_COMPARE)
- `TASK`: Task type to run - either "prompt" or "token" (optional, auto-detected based on which fields are present)

The task is automatically determined based on the configuration:
- If `DIFF_PROMPTS` or `SIM_PROMPTS` are present, it runs PROMPT_SUBGRAPH_COMPARE
- If `TOKEN_OF_INTEREST` is present, it runs TOKEN_SUBGRAPH_COMPARE
- You can explicitly set the `TASK` field to override this behavior

See the Task Selection section below for details on how these parameters are used.

## Usage

Run the program from the main script:

```bash
python main.py
```

## Task Selection

The tool supports two different analysis tasks, configured via the config file:

### 1. PROMPT_SUBGRAPH_COMPARE

Finds unique and/or overlapping features from the subgraphs of multiple different prompts.

**Configuration:**

Set the following in your config file:
- `MAIN_PROMPT`: The main memorized prompt to analyze
- `DIFF_PROMPTS`: Array of prompts where features are expected NOT to be present
- `SIM_PROMPTS`: Array of prompts where features are expected to be present

This task compares the subgraph of the main prompt against:
- **DIFF_PROMPTS**: Finds features unique to the main prompt (not present in these)
- **SIM_PROMPTS**: Finds features common across all similar prompts

### 2. TOKEN_SUBGRAPH_COMPARE

Finds unique features from a token of interest's subgraph compared to subgraphs of other top output tokens.

**Configuration:**

Set the following in your config file:
- `MAIN_PROMPT`: The main memorized prompt to analyze
- `TOKEN_OF_INTEREST`: The specific token to focus analysis on

The `TOP_K` parameter (number of top output tokens to compare against) can be configured in `main.py`.

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
