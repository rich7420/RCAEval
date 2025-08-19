#!/usr/bin/env python3
"""
Selective downloader for Hugging Face datasets/models.

Usage examples (run on another machine with larger disk):

1) List files and sizes first (dry run):
   python hf_selective_download.py \
     --repo-id Lemma-RCA-NEC/Cloud_Computing_Original \
     --repo-type dataset \
     --list-only

2) Download only a specific day (e.g., 20240124.zip) and README:
   python hf_selective_download.py \
     --repo-id Lemma-RCA-NEC/Cloud_Computing_Original \
     --repo-type dataset \
     --dest /data/lemma_cloud/Cloud_Computing_Original \
     --include-patterns README.md 20240124.zip \
     --concurrency 8

3) Download preprocessed subset similarly:
   python hf_selective_download.py \
     --repo-id Lemma-RCA-NEC/Cloud_Computing_Preprocessed \
     --repo-type dataset \
     --dest /data/lemma_cloud/Cloud_Computing_Preprocessed \
     --include-patterns README.md 20240124.zip \
     --concurrency 8

Notes:
 - This script uses HfApi repo_info(files_metadata=True) to list files, then
   downloads only matched files via hf_hub_download. This avoids fetching the
   entire snapshot when only a subset is needed.
 - If you need private repo access, set HF_TOKEN in the environment, or run
   `huggingface-cli login` beforehand.
"""

import argparse
import fnmatch
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Optional, Tuple

try:
	from huggingface_hub import HfApi, hf_hub_download
except Exception as exc:
	print("ERROR: huggingface_hub is required. Install with: pip install huggingface_hub", file=sys.stderr)
	raise


def human_readable_size(num_bytes: int) -> str:
	units = ["B", "KB", "MB", "GB", "TB"]
	size = float(num_bytes)
	idx = 0
	while size >= 1024 and idx < len(units) - 1:
		size /= 1024.0
		idx += 1
	return f"{size:.2f} {units[idx]}"


def matches_any(name: str, patterns: Iterable[str]) -> bool:
	return any(fnmatch.fnmatch(name, p) for p in patterns)


def list_repo_files_with_sizes(repo_id: str, repo_type: str) -> List[Tuple[str, Optional[int]]]:
	api = HfApi()
	info = api.repo_info(repo_id=repo_id, repo_type=repo_type, files_metadata=True)
	files = []
	for sibling in getattr(info, "siblings", []) or []:
		# sibling is a dict-like structure with keys: rfilename, size, etc.
		filename = sibling.get("rfilename") or sibling.get("path") or ""
		size = sibling.get("size")
		if filename:
			files.append((filename, size))
	return files


def selective_download(
	repo_id: str,
	repo_type: str,
	dest: str,
	include_patterns: List[str],
	exclude_patterns: List[str],
	concurrency: int,
) -> List[str]:
	os.makedirs(dest, exist_ok=True)
	files = list_repo_files_with_sizes(repo_id, repo_type)
	selected: List[Tuple[str, Optional[int]]] = []

	# Filter by include/exclude
	for path, size in files:
		if include_patterns and not matches_any(path, include_patterns):
			continue
		if exclude_patterns and matches_any(path, exclude_patterns):
			continue
		selected.append((path, size))

	downloaded_paths: List[str] = []
	if not selected:
		print(f"No files selected from {repo_id}.")
		return downloaded_paths

	print(f"Selected {len(selected)} files from {repo_id} to download:")
	for path, size in selected:
		size_str = human_readable_size(size) if size is not None else "?"
		print(f"  - {path} ({size_str})")

	def _dl(file_path: str) -> str:
		# Download into dest/ keeping relative structure
		return hf_hub_download(
			repo_id=repo_id,
			filename=file_path,
			repo_type=repo_type,
			local_dir=dest,
			local_dir_use_symlinks=False,  # ignored in recent versions
		)

	with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
		futures = {ex.submit(_dl, p): p for p, _ in selected}
		for fut in as_completed(futures):
			p = futures[fut]
			try:
				outp = fut.result()
				downloaded_paths.append(outp)
				print(f"Downloaded: {p}")
			except Exception as e:
				print(f"Failed: {p} -> {e}")

	return downloaded_paths


def main() -> None:
	parser = argparse.ArgumentParser(description="Selective Hugging Face dataset downloader")
	parser.add_argument("--repo-id", required=True, help="e.g., Lemma-RCA-NEC/Cloud_Computing_Original")
	parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"], help="Repository type")
	parser.add_argument("--dest", default="./hf_download", help="Destination directory")
	parser.add_argument("--include-patterns", nargs="*", default=["README.md"], help="Glob patterns to include (default: README.md)")
	parser.add_argument("--exclude-patterns", nargs="*", default=[], help="Glob patterns to exclude")
	parser.add_argument("--list-only", action="store_true", help="Only list files that match filters; do not download")
	parser.add_argument("--concurrency", type=int, default=8, help="Parallel downloads")

	args = parser.parse_args()

	files = list_repo_files_with_sizes(args.repo_id, args.repo_type)
	if not files:
		print(f"No files found in {args.repo_id}")
		return

	def _filtered() -> List[Tuple[str, Optional[int]]]:
		out = []
		for path, size in files:
			if args.include_patterns and not matches_any(path, args.include_patterns):
				continue
			if args.exclude_patterns and matches_any(path, args.exclude_patterns):
				continue
			out.append((path, size))
		return out

	filtered = _filtered()
	print(f"Repo: {args.repo_id} (type: {args.repo_type})")
	print(f"Destination: {os.path.abspath(args.dest)}")
	print("Matched files:")
	for p, s in filtered:
		s_str = human_readable_size(s) if s is not None else "?"
		print(f"  - {p} ({s_str})")

	if args.list_only:
		print("--list-only specified; no download performed.")
		return

	downloaded = selective_download(
		repo_id=args.repo_id,
		repo_type=args.repo_type,
		dest=args.dest,
		include_patterns=args.include_patterns,
		exclude_patterns=args.exclude_patterns,
		concurrency=args.concurrency,
	)

	print(f"Downloaded {len(downloaded)} files to {os.path.abspath(args.dest)}")


if __name__ == "__main__":
	main()


