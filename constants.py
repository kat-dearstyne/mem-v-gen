DEFAULT_SAVE_DIR = "./data/neuronpedia/"
CONFIG_BASE_DIR = "configs"
SAVE_ACT_DENSITIES_FILENAME = "neuronpedia_{model}_{submodel}_feature_act_densities.csv"
SAVE_FEATURE_FILENAME = "neuronpedia_{layer}_{index}_feature.csv"
NEURONPEDIA_API_KEY_ENV_VAR="NEURONPEDIA_API_KEY"
AVAILABLE_MODELS = {
    "gemma-2-2b": ["gemmascope-transcoder-16k", "clt-hp"],
    "qwen3-4b": ["transcoder-hp"]
}
MODEL = "gemma-2-2b"
SUBMODELS = AVAILABLE_MODELS[MODEL]
DATA_PATH = "~/data/spar-memory/neuronpedia/"
OUTPUT_DIR = "output"
TOP_K = 4
PROMPT_IDS_MEMORIZED = ["mem", "memorized vs rephrased", "memorized vs made-up", "memorized vs. random"]
PROMPT_ID_BASELINE = ["made-up", "made-up vs rephrased", "made-up vs random"]
OVERLAP_ANALYSIS_FILENAME = "overlap-metrics.csv"

# Custom color palette for visualizations
COLORS = {
    'turquoise': '#34726b',
    'light_azure': '#d4edf4',
    'pastel_orange': '#ff9e7a',
    'soft_coral': '#e8a87c',
    'dusty_teal': '#5d9a96',
    'muted_peach': '#f7c59f',
}
CUSTOM_PALETTE = [
    COLORS['turquoise'],
    COLORS['pastel_orange'],
    COLORS['light_azure'],
    COLORS['soft_coral'],
    COLORS['dusty_teal'],
    COLORS['muted_peach'],
]