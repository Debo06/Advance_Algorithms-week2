from pathlib import Path
from typing import Tuple

FIGS = Path(__file__).resolve().parents[1] / "figures"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
DATA = Path(__file__).resolve().parents[1] / "data"

def ensure_dirs() -> Tuple[Path, Path, Path]:
    for p in [FIGS, ARTIFACTS, DATA]:
        p.mkdir(parents=True, exist_ok=True)
    return FIGS, ARTIFACTS, DATA

def savefig_no_style(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
