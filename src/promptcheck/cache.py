"""On-disk response cache so re-running a suite doesn't re-bill identical calls."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .providers import Completion


class Cache:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    @staticmethod
    def key(*parts: object) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    def get(self, key: str) -> Completion | None:
        path = self.dir / f"{key}.json"
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return Completion(d["text"], d.get("latency_ms", 0.0),
                              d.get("input_tokens"), d.get("output_tokens"))
        except (OSError, ValueError, KeyError):
            return None

    def put(self, key: str, c: Completion) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"text": c.text, "latency_ms": c.latency_ms,
                              "input_tokens": c.input_tokens,
                              "output_tokens": c.output_tokens})
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, self.dir / f"{key}.json")  # atomic: safe with threads
