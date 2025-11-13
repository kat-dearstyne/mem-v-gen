import os
import json

from dotenv import load_dotenv

from common_utils import user_select_prompt, user_select_models
from constants import AVAILABLE_MODELS
from subgraph_comparisons import compare_prompt_subgraphs, compare_token_subgraphs

load_dotenv()

class Task:
    PROMPT_SUBGRAPH_COMPARE = "prompt"
    TOKEN_SUBGRAPH_COMPARE = "token"

# Load config from environment variable
config_name = os.getenv("CONFIG_NAME")
config_path = os.path.join("configs", f"{config_name}.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Load prompts from config
MAIN_PROMPT = config["MAIN_PROMPT"]
DIFF_PROMPTS = config.get("DIFF_PROMPTS", [])
SIM_PROMPTS = config.get("SIM_PROMPTS", [])
TOKEN_OF_INTEREST = config.get("TOKEN_OF_INTEREST")
SELECTED_TASK = config.get("TASK")

if DIFF_PROMPTS or SIM_PROMPTS:
    assert not TOKEN_OF_INTEREST or SELECTED_TASK, ("Both TOKEN_OF_INTEREST and DIFF_PROMPTS/SIM_PROMPTS supplied. "
                                                    "Must specify what task to perform.")
    SELECTED_TASK = SELECTED_TASK if SELECTED_TASK else Task.PROMPT_SUBGRAPH_COMPARE

elif TOKEN_OF_INTEREST:
    SELECTED_TASK = SELECTED_TASK if SELECTED_TASK else Task.TOKEN_SUBGRAPH_COMPARE

MODEL = "gemma-2-2b"
SUBMODEL = AVAILABLE_MODELS[MODEL][0]

SAVE_PATH = "~/data/spar-memory/neuronpedia/"

# ================= Compre different token subgraphs ==================
TOP_K = 4

if __name__ == "__main__":
    base_save_path = os.path.expanduser(SAVE_PATH)
    model, submodel = user_select_models(model=MODEL, submodel=SUBMODEL)
    graph_dir = os.path.join(base_save_path, "graphs")

    prompt = user_select_prompt(prompt_default=MAIN_PROMPT, graph_dir=graph_dir)

    if SELECTED_TASK == Task.PROMPT_SUBGRAPH_COMPARE:
        print(f"\nStarting run with model {model} and submodel {submodel}"
              f"\nPrompt1: '{prompt}'\n"
              f"\nContrasting {len(DIFF_PROMPTS)} prompts.\n"
              f"\nComparing {len(SIM_PROMPTS)} prompts.\n")

        compare_prompt_subgraphs(main_prompt=prompt, diff_prompts=DIFF_PROMPTS, sim_prompts=SIM_PROMPTS,
                                 model=model, submodel=submodel, graph_dir=graph_dir)
    elif SELECTED_TASK == Task.TOKEN_SUBGRAPH_COMPARE:
        print(f"\nStarting run with model {model} and submodel {submodel}"
              f"\nPrompt1: '{prompt}'\n"
              f"\nToken of interest: {TOKEN_OF_INTEREST}.\n")

        compare_token_subgraphs(main_prompt=prompt, token_of_interest=TOKEN_OF_INTEREST, model=model, submodel=submodel,
                                graph_dir=graph_dir, tok_k_outputs=TOP_K)
