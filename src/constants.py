from pathlib import Path

from src.utils import get_env_bool

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

"""
========= Paths ===========
"""
DEFAULT_SAVE_DIR = PROJECT_ROOT / "data" / "neuronpedia"
CONFIG_BASE_DIR = PROJECT_ROOT / "configs"
DATA_PATH = "~/data/spar-memory/neuronpedia/"
OUTPUT_DIR = PROJECT_ROOT / "output"

"""
========= Filenames ===========
"""
SAVE_ACT_DENSITIES_FILENAME = "neuronpedia_{model}_{submodel}_feature_act_densities.csv"
SAVE_FEATURE_FILENAME = "neuronpedia_{layer}_{index}_feature.csv"
OVERLAP_ANALYSIS_FILENAME = "overlap-metrics.csv"

"""
========= Models ===========
"""
AVAILABLE_MODELS = {
    "gemma-2-2b": ["gemmascope-transcoder-16k", "clt-hp"],
    "qwen3-4b": ["transcoder-hp"]
}
MODEL = "gemma-2-2b"
SUBMODELS = AVAILABLE_MODELS[MODEL]

"""
========= Random ===========
"""
MIN_ACTIVATION_DENSITY=30
PROMPT_IDS_MEMORIZED = ["memorized", "rephrased", "made-up", "random"]
PROMPT_ID_BASELINE = ["made-up", "made-up vs rephrased", "made-up vs random"]
TOP_K = 4
FEATURE_LAYER = "25"
FEATURE_ID = "9031"

"""
========= Environment ===========
"""
IS_TEST = get_env_bool("IS_TEST", False)


"""
========= Visualization ===========
"""
COLORS = {
    'turquoise': '#34726b',
    'light_azure': '#d4edf4',
    'pastel_orange': '#ff9e7a',
    'soft_coral': '#e8a87c',
    'dusty_teal': '#5d9a96',
    'muted_peach': '#f7c59f',
    'slate_blue': '#6b7a8f',
    'warm_sand': '#d4a574',
    'sage_green': '#8fbc8f',
    'dusty_rose': '#c9a9a6',
    'steel_blue': '#4682b4',
    'terracotta': '#cc7a6f',
    'olive_green': '#6b8e23',
    'mauve': '#b784a7',
    'golden_rod': '#daa520',
}
CUSTOM_PALETTE = [
    COLORS['turquoise'],
    COLORS['pastel_orange'],
    COLORS['dusty_teal'],
    COLORS['soft_coral'],
    COLORS['slate_blue'],
    COLORS['warm_sand'],
    COLORS['sage_green'],
    COLORS['dusty_rose'],
    COLORS['steel_blue'],
    COLORS['terracotta'],
    COLORS['olive_green'],
    COLORS['mauve'],
    COLORS['golden_rod'],
    COLORS['light_azure'],
    COLORS['muted_peach'],
]