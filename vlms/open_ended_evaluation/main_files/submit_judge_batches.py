#!/usr/bin/env python3
"""Submit both judge JSONL files once and persist provider batch identifiers."""

import argparse
import json
import os
from pathlib import Path

from google import genai
from google.genai import types
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", type=Path, required=True)
    args = parser.parse_args()
    state_path = args.batch_dir / "batch_state.json"
    metadata = json.loads((args.batch_dir / "request_metadata.json").read_text())
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(metadata)

    def save_state():
        state_path.write_text(json.dumps(state, indent=2))

    if not state.get("openai", {}).get("batch_id"):
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        with (args.batch_dir / "openai_requests.jsonl").open("rb") as handle:
            openai_file = openai_client.files.create(file=handle, purpose="batch")
        state["openai"] = {"input_file_id": openai_file.id, "status": "uploaded"}
        save_state()
        openai_batch = openai_client.batches.create(
            input_file_id=openai_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"experiment": "guic_open_ended_vqa", "judge": "openai"},
        )
        state["openai"].update({"batch_id": openai_batch.id, "status": openai_batch.status})
        save_state()
        print(f"[SUBMITTED] provider=openai file_id={openai_file.id} batch_id={openai_batch.id}")
    else:
        print(f"[SKIP] provider=openai existing_batch={state['openai']['batch_id']}")

    if not state.get("gemini", {}).get("batch_name"):
        gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        gemini_file = gemini_client.files.upload(
            file=args.batch_dir / "gemini_requests.jsonl",
            config=types.UploadFileConfig(display_name="guic-open-ended-judge", mime_type="jsonl"),
        )
        state["gemini"] = {"input_file_name": gemini_file.name, "status": "uploaded"}
        save_state()
        gemini_batch = gemini_client.batches.create(
            model=metadata["gemini_model"],
            src=gemini_file.name,
            config={"display_name": "guic-open-ended-semantic-judge"},
        )
        state["gemini"].update({"batch_name": gemini_batch.name, "status": str(gemini_batch.state)})
        save_state()
        print(f"[SUBMITTED] provider=gemini file={gemini_file.name} batch={gemini_batch.name}")
    else:
        print(f"[SKIP] provider=gemini existing_batch={state['gemini']['batch_name']}")

    print(f"[PASS] saved_state={state_path}")


if __name__ == "__main__":
    main()
