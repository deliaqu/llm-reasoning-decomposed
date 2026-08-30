import os as _os
HOME_DIR = _os.path.dirname(_os.path.abspath(__file__))
DATA_DIR = f"{HOME_DIR}/data"
RESULT_DIR = f"{HOME_DIR}/results/disentangled_evaluation"

CACHE_DIR = _os.environ.get("CACHE_DIR", _os.path.join(HOME_DIR, "cache"))

BEHAVIOUR_PLOT_DIR = f"{HOME_DIR}/results/disentangled_evaluation/plots"


