"""Utilities for instruction-conditioned profiling.

This project extension treats an 'instruction' as an additional prefix that is prepended to every
input context before it is fed into the embedding model (encoder or decoder).

For encoder-style models (BERT/ALBERT/DistilBERT), this is the most practical way to model
instruction/persona conditioning while keeping the rest of the pipeline unchanged.

The module provides:
- loading instructions from JSON/TXT
- applying an instruction prefix to a context
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, Iterable, Tuple, Optional


@dataclass(frozen=True)
class Instruction:
    """A named instruction/persona."""
    name: str
    text: str


DEFAULT_INSTRUCTION_TEMPLATE = "Instruction: {instruction}\nText: {context}"


def apply_instruction_to_context(
    context: str,
    instruction_text: str | None,
    template: str = DEFAULT_INSTRUCTION_TEMPLATE,
) -> str:
    """Return a new context string with an instruction prefix.

    If instruction_text is None or empty, returns context unchanged.
    """
    if instruction_text is None:
        return context
    instruction_text = instruction_text.strip()
    if instruction_text == "":
        return context
    return template.format(instruction=instruction_text, context=context)


def load_instructions(path: str | Path) -> Dict[str, str]:
    """Load instructions from a JSON or TXT file.

    Supported formats:
    - JSON: {"baseline": "", "conservative": "...", "liberal": "..."}
    - TXT: one instruction per line.
        * Either:  name\ttext
        * Or:      text  (name auto-generated as instr_000, instr_001, ...)

    Returns: dict {name: text}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Instruction file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON instructions must be an object/dict mapping name->text.")
        return {str(k): ("" if v is None else str(v)) for k, v in data.items()}

    # TXT fallback
    instructions: Dict[str, str] = {}
    lines = [ln.strip("\n") for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    auto_i = 0
    for ln in lines:
        if "\t" in ln:
            name, text = ln.split("\t", 1)
            name = name.strip()
            text = text.strip()
        else:
            name = f"instr_{auto_i:03d}"
            text = ln.strip()
            auto_i += 1
        if name in instructions:
            raise ValueError(f"Duplicate instruction name '{name}' in {path}")
        instructions[name] = text
    return instructions
