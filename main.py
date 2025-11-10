import os

from dotenv import load_dotenv

from common_utils import user_select_prompt, user_select_models
from subgraph_comparisons import compare_prompt_subgraphs, compare_token_subgraphs

load_dotenv()

class Task:
    PROMPT_SUBGRAPH_COMPARE = "prompt"
    TOKEN_SUBGRAPH_COMPARE = "token"

SELECTED_TASK = Task.TOKEN_SUBGRAPH_COMPARE

# MEMORIZED PROMPT OF INTEREST
MAIN_PROMPT = "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF"
MODEL = "gemma-2-2b"
SUBMODEL = "clt-hp"

SAVE_PATH = "~/data/spar-memory/neuronpedia/"

# ================= Compare different prompt subgraphs ===============
# Prompts where feature is expected to NOT be in any
DIFF_PROMPTS = [
    "THE SOFTWARE AND ANY ACCOMPANYING MATERIALS ARE PROVIDED \"AS IS\", "
    "WITHOUT ANY PROMISE OR GUARANTEE OF PERFORMANCE, RELIABILITY, OR SUITABILITY AND THE WARRANTIES OF",
    "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT"
]
# Prompts where feature is expected to be present in all
SIM_PROMPTS = [
    "The software is provided \"as is\", without warranty of any kind, express or implied, "
    "including but not limited to the warranties of"
]

# ================= Compre different token subgraphs ==================
TOKEN_OF_INTEREST = "\""
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
