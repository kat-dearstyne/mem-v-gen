from dotenv import load_dotenv

from src.constants import OUTPUT_DIR
from src.analysis_runner import (
    run_for_all_configs,
    run_cross_condition_analysis,
    analyze_conditions_post_run
)
from src.utils import get_env_bool, get_env_list, get_env_int_list, get_env_grouped_list

load_dotenv()

# ================= Setup Variables ==================
CONFIG_NAMES = get_env_list("CONFIG_NAME")
CONFIG_DIRS = get_env_list("CONFIG_DIRS")
RUN_ERROR_ANALYSIS = get_env_bool("RUN_ERROR_ANALYSIS", False)
ANALYSIS_DIRS = get_env_list("ANALYSIS_DIRS")
RUN_FINAL_ANALYSIS = get_env_bool("RUN_FINAL_ANALYSIS", False) and ANALYSIS_DIRS
SUBMODEL_NUMS = get_env_int_list("SUBMODEL_NUMS", default=[0])
# PROMPT_IDS can be: "id1,id2,id3" (same for all conditions) or "id1a,id1b;id2a,id2b" (per condition)
PROMPT_IDS_GROUPS = get_env_grouped_list("PROMPT_IDS")

if __name__ == "__main__":
    if RUN_FINAL_ANALYSIS:
        assert len(ANALYSIS_DIRS) >= 2, f"Expected at least 2 dirs for analysis. Got {len(ANALYSIS_DIRS)}"
        dirs = [OUTPUT_DIR / d.strip() for d in ANALYSIS_DIRS if d.strip()]
        analyze_conditions_post_run(dirs=dirs, save_dir=dirs[0] / "final")
    elif len(CONFIG_DIRS) > 1 or len(SUBMODEL_NUMS) > 1:
        # Cross-condition analysis across multiple config dirs or submodels
        run_cross_condition_analysis(
            config_dirs=CONFIG_DIRS if CONFIG_DIRS else None,
            config_names=CONFIG_NAMES if CONFIG_NAMES else None,
            submodel_nums=SUBMODEL_NUMS,
            run_error_analysis=RUN_ERROR_ANALYSIS,
            prompt_ids_per_condition=PROMPT_IDS_GROUPS if PROMPT_IDS_GROUPS else None
        )
    else:
        # Single condition run - use first group if available
        prompt_ids = PROMPT_IDS_GROUPS[0] if PROMPT_IDS_GROUPS else None
        run_for_all_configs(CONFIG_NAMES, CONFIG_DIRS[0] if CONFIG_DIRS else "",
                            RUN_ERROR_ANALYSIS, submodel_num=SUBMODEL_NUMS[0],
                            prompt_ids=prompt_ids)
