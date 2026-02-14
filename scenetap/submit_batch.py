import argparse
import time
import requests
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def submit_batch(client, batch_file_path, description="MCQ generation batch", endpoint="/v1/chat/completions", completion_window="24h"):
    print(f"📤 Uploading file: {batch_file_path}")
    uploaded = client.files.create(
        file=open(batch_file_path, "rb"),
        purpose="batch"
    )
    input_file_id = uploaded.id
    print(f"✅ File uploaded. File ID: {input_file_id}")

    print("🚀 Submitting batch job...")
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata={"description": description}
    )
    print(f"📦 Batch submitted. Batch ID: {batch.id}")
    return batch.id

def watch_batch(client, batch_id, interval=15):
    print(f"⏳ Watching batch {batch_id} (checking every {interval}s)")
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"🕒 Status: {batch.status}")
        if batch.status in ["completed", "failed", "cancelled", "expired"]:
            return batch
        time.sleep(interval)

def download_output_file(client, batch, output_path):
    # output_file_id = "file-Xw7MbThPBmBsiAC5zDtyXW"
    output_file_id = batch.output_file_id
    if not output_file_id:
        print("❌ No output file ID found. Batch likely failed.")
        return

    print(f"📦 Output File ID: {output_file_id} – downloading content...")

    # Download file content using .content(file_id).read()
    content_stream = client.files.content(output_file_id)
    content = content_stream.read()

    with open(output_path, "wb") as f:
        f.write(content)

    print(f"✅ Output downloaded successfully to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Submit and monitor OpenAI Batch job")
    parser.add_argument("--input", default = './batches_files_validation/batch_file_validation_3000_4000.jsonl', help="Path to your batch JSONL input file")
    parser.add_argument("--output", required=False, default="batch_output.jsonl", help="Output file to save results")
    parser.add_argument("--interval", type=int, default=15, help="Polling interval in seconds")
    parser.add_argument("--description", type=str, default="generate plausible answers for the given questions", help="Batch job description")
    parser.add_argument("--completion_window", type=str, default="24h", help="OpenAI completion window")
    args = parser.parse_args()

    # Ensure the API key is set
    import os
    api_key = os.getenv("OPENAI_API_KEY")


    if not api_key:
        raise EnvironmentError("Please set your OPENAI_API_KEY as an environment variable.")

    # # Instantiate client
    client = OpenAI(api_key=api_key)

    # # Submit, watch, and download
    batch_id = submit_batch(client, args.input, args.description, completion_window=args.completion_window)
    final_batch = watch_batch(client, batch_id, interval=args.interval)
    download_output_file(client, final_batch, args.output)

    # batch_id = "batch_6850811d5e4c81909509fe825451523f"
    # batch = client.batches.retrieve(batch_id)
    # error_file_id = batch.error_file_id
    # print(batch)
    # if error_file_id:
    #     print(f"❗ Batch failed with error file ID: {error_file_id}")
    #     error_content = client.files.content(error_file_id).read()
    #     with open("error_log.txt", "wb") as f:
    #         f.write(error_content)
    #     print("Error details saved to error_log.txt")
    # else:
    #     print("Batch completed successfully, no errors found.")

if __name__ == "__main__":
    main()
