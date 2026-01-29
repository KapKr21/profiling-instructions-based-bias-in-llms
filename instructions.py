from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, Iterable, Tuple, Optional

@dataclass(frozen=True)
class Instruction:
    """A named instruction."""
    name: str
    text: str

DEFAULT_INSTRUCTION_TEMPLATE = "Instruction: {instruction}\nText: {context}"

def apply_instruction_to_context(context: str, 
                                 instruction_text: str | None,
                                 template: str = DEFAULT_INSTRUCTION_TEMPLATE,) -> str:
    if instruction_text is None:
        return context
    
    instruction_text = instruction_text.strip()

    if instruction_text == "":
        return context
    
    return template.format(instruction=instruction_text, 
                           context=context)

def load_instructions(path: str | Path) -> Dict[str, str]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Instruction file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON instructions must be an object/dict mapping name->text.")
        return {str(k): ("" if v is None else str(v)) for k, v in data.items()}

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
