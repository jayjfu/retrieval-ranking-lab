"""Average the LoRA weights of several checkpoints of one run into a new adapter."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer
from safetensors.torch import load_file, save_file

app = typer.Typer(add_completion=False)
ADAPTER_FILE = "adapter_model.safetensors"


@app.command()
def main(
    checkpoint: list[Path] = typer.Option(..., help="checkpoint directories to average, repeatable"),
    out: Path = typer.Option(..., help="e.g. checkpoints/reranker/<run>/average_last3"),
):
    weights = [load_file(path / ADAPTER_FILE) for path in checkpoint]
    if any(state.keys() != weights[0].keys() for state in weights):
        raise typer.BadParameter("checkpoints must come from the same LoRA configuration")
    averaged = {name: sum(state[name].float() for state in weights).div(len(weights)).to(weights[0][name].dtype)
                for name in weights[0]}
    out.mkdir(parents=True, exist_ok=True)
    for file in checkpoint[-1].iterdir():  # adapter config and tokenizer only, not run logs
        if file.is_file() and (file.name == "adapter_config.json" or file.name.startswith(("tokenizer", "chat_template"))):
            shutil.copy(file, out / file.name)
    save_file(averaged, out / ADAPTER_FILE)
    (out / "averaged_from.json").write_text(json.dumps([str(path) for path in checkpoint], indent=2))
    typer.echo(f"averaged {len(checkpoint)} checkpoints -> {out}")


if __name__ == "__main__":
    app()
