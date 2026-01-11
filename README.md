# mem-v-gen

## Installation

1. Install dependencies from the requirements file:

```bash
pip install -r requirement.txt
```

2. Create a `.env` file in the project root:

```bash
NEURONPEDIA_API_KEY=your_api_key_here
HF_TOKEN=your_hf_token_here    # Optional: Required for replacement model tasks
CONFIG_NAME=declaration,dream  # Comma-separated config names or "all" for every config in directory
CONFIG_DIRS=                   # Optional: subdirectory in configs/
TASK=prompt                    # Optional: Task type (see Task Selection)
SUBMODEL_NUMS=0,1              # Optional: Submodel indices for cross-condition analysis
PROMPT_IDS=id1,id2,id3         # Optional: Neuronpedia graph IDs for each prompt
RESULTS_DIR=output             # Optional: Directory for saving results (default: output)
```

## Configuration Files

Prompt configurations are stored as JSON files in the `configs/` directory.

**Required:**
- `MAIN_PROMPT`: The main memorized prompt to analyze

**Optional (task-dependent):**
- `DIFF_PROMPTS`: Contrasting prompts (for `prompt` task)
- `SIM_PROMPTS`: Similar prompts (for `prompt` task)
- `TOKEN_OF_INTEREST`: Token to analyze (for `token` task)
- `MEMORIZED_COMPLETION`: Expected completion token (for accuracy metrics)
- `PROMPT_IDS`: Graph IDs mapping (can also be set via env var)
- `TASK`: Task type (auto-detected from fields if not specified)

Example config:
```json
{
  "MAIN_PROMPT": "When in the Course of human events...",
  "DIFF_PROMPTS": ["Alternative prompt 1", "Alternative prompt 2"],
  "SIM_PROMPTS": [],
  "MEMORIZED_COMPLETION": "earth"
}
```

## Usage

Run the program from the main script:

```bash
python main.py
```

## Task Selection

Tasks are configured via the `TASK` environment variable or config file:

| Task | Value | Description                                    |
|------|-------|------------------------------------------------|
| PROMPT_SUBGRAPH_COMPARE | `prompt` | Compare subgraphs across multiple prompts      |
| TOKEN_SUBGRAPH_COMPARE | `token` | Compare token subgraphs within a single prompt |
| FEATURE_OVERLAP | `feature_overlap` | Analyze feature overlap across prompts         |
| REPLACEMENT_MODEL | `replacement_model` | Test transcoder reconstruction accuracy        |
| EARLY_LAYER_CONTRIBUTION | `early_layer` | Measure early layer feature contributions      |
| L0_REPLACEMENT_MODEL | `l0` | Compute L0 (active features) per layer         |

Tasks are auto-detected if not specified:
- If `DIFF_PROMPTS` or `SIM_PROMPTS` present → `prompt`
- If `TOKEN_OF_INTEREST` present → `token`

## Output

Results are saved to the `output/` directory. 
The program prints Neuronpedia URLs for attribution graphs where applicable. 
