import json
import sys
import time
from pathlib import Path


STATUS_PATH = Path("logs/status.json")


class RunLog:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def event(self, kind: str, **fields) -> None:
        rec = {"t": round(time.time() - self.t0, 1), "kind": kind, **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        STATUS_PATH.write_text(json.dumps({
            "phase": self.path.stem,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": rec["t"],
            "last_event": rec,
        }, indent=1))


def eta_str(seconds: float) -> str:
    m, s = divmod(int(max(seconds, 0)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def show_progress_bars() -> bool:
    return sys.stderr.isatty()
