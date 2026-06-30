#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import torch


def numpy_to_torch_dtype(array: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy array to a Torch tensor while preserving shape and dtype
    whenever PyTorch supports that NumPy dtype directly.
    """
    return torch.from_numpy(array)


def convert_npy_file(npy_path: Path, overwrite: bool = False) -> None:
    pt_path = npy_path.with_suffix(".pt")

    if pt_path.exists() and not overwrite:
        print(f"skip: {pt_path} already exists")
        return

    array = np.load(npy_path, allow_pickle=False)
    tensor = numpy_to_torch_dtype(array)

    torch.save(tensor, pt_path)

    print(
        f"converted: {npy_path} -> {pt_path} "
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
    )


def convert_tree(root: Path, overwrite: bool = False) -> None:
    npy_files = sorted(root.rglob("*.npy"))

    if not npy_files:
        print(f"No .npy files found under: {root}")
        return

    for npy_path in npy_files:
        convert_npy_file(npy_path, overwrite=overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively convert NumPy .npy files to PyTorch .pt tensors."
    )

    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to search recursively. Default: current directory.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files.",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    convert_tree(root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
