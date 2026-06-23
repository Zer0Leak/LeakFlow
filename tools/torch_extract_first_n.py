#!/usr/bin/env python3

from pathlib import Path
import argparse
import torch


def extract_first_n(input_path: Path, n: int, overwrite: bool = False) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"File does not exist: {input_path}")

    if not input_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {input_path}")

    tensor = torch.load(input_path, map_location="cpu")

    if not torch.is_tensor(tensor):
        raise TypeError(f"File does not contain a torch.Tensor: {input_path}")

    if tensor.ndim == 0:
        raise ValueError(f"Cannot slice axis 0 of a scalar tensor: {input_path}")

    extracted = tensor[:n]

    if overwrite:
        output_path = input_path
    else:
        output_path = input_path.with_name(
            f"{input_path.stem}_first_{n}{input_path.suffix}"
        )

    torch.save(extracted, output_path)

    print(
        f"saved: {output_path} "
        f"original_shape={tuple(tensor.shape)} "
        f"new_shape={tuple(extracted.shape)} "
        f"dtype={extracted.dtype}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract first N entries along axis/dim 0 from one .pt tensor file."
    )

    parser.add_argument(
        "-n",
        "--num",
        type=int,
        required=True,
        help="Number of entries to keep from axis/dim 0.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the original file.",
    )

    parser.add_argument(
        "file",
        help="Input .pt tensor file. This must be the last argument.",
    )

    args = parser.parse_args()

    if args.num <= 0:
        raise ValueError("N must be greater than zero")

    input_path = Path(args.file).expanduser().resolve()

    extract_first_n(
        input_path=input_path,
        n=args.num,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
