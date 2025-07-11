import os
from typing import Optional

from requests.exceptions import HTTPError


def hf_download(out_dir: str, repo_id: Optional[str] = None, hf_token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download
    os.makedirs(f"{out_dir}", exist_ok=True)
    try:
        snapshot_download(repo_id, local_dir=f"{out_dir}", local_dir_use_symlinks=False, token=hf_token)
    except HTTPError as e:
        if e.response.status_code == 401:
            print("You need to pass a valid `--hf_token=...` to download private checkpoints.")
        else:
            raise e

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download data from HuggingFace Hub.')
    parser.add_argument('--repo_id', type=str, default="checkpoints/meta-llama/llama-2-7b-chat-hf", help='Repository ID to download from.')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace API token.')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory to save the downloaded data.')

    args = parser.parse_args()
    hf_download(args.out_dir, args.repo_id, args.hf_token)