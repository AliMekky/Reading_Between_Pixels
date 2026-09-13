#!/usr/bin/env python3
"""Check both asynchronous judges and download results when available."""

import argparse
import json
import os
from pathlib import Path

from google import genai
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", type=Path, required=True)
    args = parser.parse_args()
    state_path = args.batch_dir / "batch_state.json"
    state = json.loads(state_path.read_text())

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    openai_batch = openai_client.batches.retrieve(state["openai"]["batch_id"])
    state["openai"]["status"] = openai_batch.status
    state["openai"]["request_counts"] = (
        openai_batch.request_counts.model_dump() if openai_batch.request_counts else None
    )
    if openai_batch.output_file_id:
        output = openai_client.files.content(openai_batch.output_file_id).content
        (args.batch_dir / "openai_results.jsonl").write_bytes(output)
        state["openai"]["output_file_id"] = openai_batch.output_file_id
    if openai_batch.error_file_id:
        errors = openai_client.files.content(openai_batch.error_file_id).content
        (args.batch_dir / "openai_errors.jsonl").write_bytes(errors)
        state["openai"]["error_file_id"] = openai_batch.error_file_id
    print(f"[STATUS] provider=openai status={openai_batch.status} counts={state['openai']['request_counts']}")

    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    gemini_batch = gemini_client.batches.get(name=state["gemini"]["batch_name"])
    state["gemini"]["status"] = str(gemini_batch.state)
    stats = getattr(gemini_batch, "completion_stats", None)
    state["gemini"]["stats"] = stats.model_dump(mode="json") if stats else None
    destination = getattr(gemini_batch, "dest", None)
    output_file = getattr(destination, "file_name", None) if destination else None
    if output_file:
        gemini_client.files.download(file=output_file, destination=args.batch_dir / "gemini_results.jsonl")
        state["gemini"]["output_file_name"] = output_file
    print(f"[STATUS] provider=gemini status={gemini_batch.state} stats={state['gemini']['stats']}")

    state_path.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
