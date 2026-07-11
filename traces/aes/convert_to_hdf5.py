#!/usr/bin/env python3
"""Convert LeakFlow AES key folders into verified, atomic HDF5 files."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Populated by load_dependencies() so --help and --dry-run work without the
# optional conversion stack installed.
np: Any = None
torch: Any = None
h5py: Any = None
cryptography: Any = None
Cipher: Any = None
algorithms: Any = None
modes: Any = None


SCHEMA_NAME = "leakflow.sca.tensor-dataset"
SCHEMA_VERSION = 1
CONVERTER_VERSION = 1
DEFAULT_CHUNK_TARGET_BYTES = 1024 * 1024
DEFAULT_IO_ROWS = 256
KEY_DIRECTORY_PATTERN = re.compile(r"^key_[0-9]+$")


class ConversionError(RuntimeError):
    """A conversion error that is safe to present without a traceback."""


def load_dependencies() -> None:
    """Load optional converter dependencies after argparse handles --help."""
    modules: dict[str, str] = {
        "numpy": "numpy",
        "torch": "torch",
        "h5py": "h5py",
        "cryptography": "cryptography",
    }
    loaded: dict[str, Any] = {}
    failures: list[str] = []

    for package, module in modules.items():
        try:
            loaded[package] = importlib.import_module(module)
        except (ImportError, OSError) as error:
            failures.append(f"{package}: {error}")

    if failures:
        packages = " ".join(modules)
        raise ConversionError(
            "Could not load the converter dependency stack:\n  "
            + "\n  ".join(failures)
            + "\nInstall compatible packages with:\n"
            f"  python3 -m pip install {packages}"
        )

    globals().update(loaded)
    globals()["np"] = loaded["numpy"]
    ciphers = importlib.import_module("cryptography.hazmat.primitives.ciphers")
    globals()["Cipher"] = ciphers.Cipher
    globals()["algorithms"] = ciphers.algorithms
    globals()["modes"] = ciphers.modes


@dataclass(frozen=True)
class StorageConfig:
    compression: str
    compression_level: int
    shuffle: bool
    fletcher32: bool
    chunk_target_bytes: int
    io_rows: int
    created_utc: str


@dataclass(frozen=True)
class SourceArrays:
    traces: Any
    plaintexts: Any
    key: Any

    @property
    def trace_count(self) -> int:
        return int(self.traces.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.traces.shape[1])


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    shape: tuple[int, ...]
    dtype: str
    axes: str
    role: str
    chunked: bool = True
    row_aligned: bool = True


class WriteProgress:
    def __init__(self, label: str, total_bytes: int) -> None:
        self.label = label
        self.total_bytes = max(total_bytes, 1)
        self.completed_bytes = 0
        self.last_percent = -5
        self.last_report = 0.0

    def advance(self, byte_count: int, operation: str) -> None:
        self.completed_bytes += byte_count
        percent = min(100, int(self.completed_bytes * 100 / self.total_bytes))
        now = time.monotonic()
        if (
            percent >= self.last_percent + 5
            or percent == 100
            or now - self.last_report >= 1.0
        ):
            print(
                f"  {self.label}: {percent:3d}% "
                f"({self.completed_bytes}/{self.total_bytes} logical bytes) {operation}",
                flush=True,
            )
            self.last_percent = percent
            self.last_report = now


def sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def torch_load_tensor(path: Path) -> Any:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise ConversionError(
            f"{path}: this converter requires a PyTorch release that supports "
            "torch.load(..., weights_only=True)"
        ) from error
    except Exception as error:
        raise ConversionError(f"Could not load Torch tensor {path}: {error}") from error

    if not isinstance(value, torch.Tensor):
        raise ConversionError(
            f"{path}: expected a Torch tensor, got {type(value).__name__}"
        )
    return value.detach().cpu().contiguous()


def load_and_validate_source(key_directory: Path) -> SourceArrays:
    paths = {
        "traces": key_directory / "traces.pt",
        "plaintexts": key_directory / "plain_texts.pt",
        "key": key_directory / "key.pt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ConversionError(
            f"{key_directory}: missing required input file(s): {', '.join(missing)}"
        )

    print(f"  {key_directory.name}: loading source tensors", flush=True)
    traces_tensor = torch_load_tensor(paths["traces"])
    plaintexts_tensor = torch_load_tensor(paths["plaintexts"])
    key_tensor = torch_load_tensor(paths["key"])

    if traces_tensor.dtype != torch.float32 or traces_tensor.ndim != 2:
        raise ConversionError(
            f"{paths['traces']}: expected float32 [N,5000], got "
            f"{traces_tensor.dtype} {tuple(traces_tensor.shape)}"
        )
    if tuple(traces_tensor.shape[1:]) != (5000,):
        raise ConversionError(
            f"{paths['traces']}: expected 5000 samples per trace, got "
            f"{tuple(traces_tensor.shape)}"
        )
    if plaintexts_tensor.dtype != torch.uint8 or plaintexts_tensor.ndim != 2:
        raise ConversionError(
            f"{paths['plaintexts']}: expected uint8 [N,16], got "
            f"{plaintexts_tensor.dtype} {tuple(plaintexts_tensor.shape)}"
        )
    if tuple(plaintexts_tensor.shape[1:]) != (16,):
        raise ConversionError(
            f"{paths['plaintexts']}: expected 16-byte plaintexts, got "
            f"{tuple(plaintexts_tensor.shape)}"
        )
    if key_tensor.dtype != torch.uint8 or tuple(key_tensor.shape) != (16,):
        raise ConversionError(
            f"{paths['key']}: expected uint8 [16], got "
            f"{key_tensor.dtype} {tuple(key_tensor.shape)}"
        )
    if traces_tensor.shape[0] <= 0:
        raise ConversionError(f"{key_directory}: the input contains no traces")
    if traces_tensor.shape[0] != plaintexts_tensor.shape[0]:
        raise ConversionError(
            f"{key_directory}: trace/plaintext row count mismatch "
            f"({traces_tensor.shape[0]} != {plaintexts_tensor.shape[0]})"
        )

    return SourceArrays(
        traces=traces_tensor.numpy(),
        plaintexts=plaintexts_tensor.numpy(),
        key=key_tensor.numpy(),
    )


def canonical_array(array: Any, dtype: str) -> Any:
    return np.ascontiguousarray(array, dtype=np.dtype(dtype))


def chunk_shape(
    shape: tuple[int, ...], dtype: str, target_bytes: int
) -> tuple[int, ...]:
    row_items = 1
    for extent in shape[1:]:
        row_items *= extent
    row_bytes = max(1, row_items * np.dtype(dtype).itemsize)
    rows = max(1, min(shape[0], target_bytes // row_bytes))
    return (rows, *shape[1:])


def chunked_dataset_options(spec: DatasetSpec, config: StorageConfig) -> dict[str, Any]:
    options: dict[str, Any] = {
        "chunks": chunk_shape(spec.shape, spec.dtype, config.chunk_target_bytes),
        "shuffle": config.shuffle,
        "fletcher32": config.fletcher32,
    }
    if config.compression == "gzip":
        options["compression"] = "gzip"
        options["compression_opts"] = config.compression_level
    return options


def stamp_dataset_attributes(
    dataset: Any,
    spec: DatasetSpec,
    logical_sha256: str,
    logical_nbytes: int,
    config: StorageConfig,
) -> None:
    dataset.attrs["tensor.axes"] = spec.axes
    dataset.attrs["origin.role"] = spec.role
    dataset.attrs["leakflow.logical_sha256"] = logical_sha256
    dataset.attrs["leakflow.logical_nbytes"] = logical_nbytes
    dataset.attrs["leakflow.row_aligned"] = spec.row_aligned
    dataset.attrs["origin.storage.layout"] = "chunked" if spec.chunked else "contiguous"
    dataset.attrs["origin.storage.chunk_shape"] = (
        ",".join(str(value) for value in dataset.chunks) if dataset.chunks else ""
    )
    dataset.attrs["origin.storage.compression"] = dataset.compression or "none"
    dataset.attrs["origin.storage.compression_level"] = (
        int(dataset.compression_opts) if dataset.compression == "gzip" else 0
    )
    dataset.attrs["origin.storage.shuffle"] = bool(dataset.shuffle)
    dataset.attrs["origin.storage.checksum"] = (
        "fletcher32" if dataset.fletcher32 else "none"
    )
    if spec.role == "traces":
        dataset.attrs["payload.leakage.inverted"] = False


def write_array_dataset(
    h5_file: Any,
    spec: DatasetSpec,
    source: Any,
    config: StorageConfig,
    progress: WriteProgress,
) -> str:
    options = chunked_dataset_options(spec, config) if spec.chunked else {}
    dataset = h5_file.create_dataset(
        spec.path, shape=spec.shape, dtype=spec.dtype, **options
    )
    digest = hashlib.sha256()

    if spec.chunked:
        for begin in range(0, spec.shape[0], config.io_rows):
            end = min(begin + config.io_rows, spec.shape[0])
            block = canonical_array(source[begin:end], spec.dtype)
            dataset[begin:end] = block
            raw = block.tobytes(order="C")
            digest.update(raw)
            progress.advance(len(raw), f"writing {spec.path}")
    else:
        block = canonical_array(source, spec.dtype)
        dataset[...] = block
        raw = block.tobytes(order="C")
        digest.update(raw)
        progress.advance(len(raw), f"writing {spec.path}")

    result = digest.hexdigest()
    stamp_dataset_attributes(
        dataset,
        spec,
        result,
        int(np.prod(spec.shape)) * np.dtype(spec.dtype).itemsize,
        config,
    )
    if spec.path == "/keys":
        dataset.attrs["payload.crypto.key.scope"] = "fixed-per-file"
    return result


def write_ciphertexts(
    h5_file: Any,
    spec: DatasetSpec,
    plaintexts: Any,
    key: Any,
    config: StorageConfig,
    progress: WriteProgress,
) -> str:
    dataset = h5_file.create_dataset(
        spec.path,
        shape=spec.shape,
        dtype=spec.dtype,
        **chunked_dataset_options(spec, config),
    )
    digest = hashlib.sha256()
    encryptor = Cipher(algorithms.AES(bytes(key)), modes.ECB()).encryptor()

    for begin in range(0, spec.shape[0], config.io_rows):
        end = min(begin + config.io_rows, spec.shape[0])
        plaintext_block = canonical_array(plaintexts[begin:end], "u1")
        encrypted = encryptor.update(plaintext_block.tobytes(order="C"))
        ciphertext_block = np.frombuffer(encrypted, dtype=np.uint8).reshape(
            end - begin, 16
        )
        dataset[begin:end] = ciphertext_block
        digest.update(encrypted)
        progress.advance(len(encrypted), f"deriving {spec.path}")

    if encryptor.finalize():
        raise ConversionError(
            "AES encryptor produced an unexpected final partial block"
        )

    result = digest.hexdigest()
    stamp_dataset_attributes(dataset, spec, result, spec.shape[0] * 16, config)
    dataset.attrs["payload.crypto.ciphertext.provenance"] = "derived"
    dataset.attrs["payload.crypto.ciphertext.derivation"] = (
        "aes-128-encrypt(plaintext,key)"
    )
    return result


def write_jitter_iterations(
    h5_file: Any,
    spec: DatasetSpec,
    plaintexts: Any,
    config: StorageConfig,
    progress: WriteProgress,
) -> str:
    jitter = h5_file.require_group("/countermeasures/jitter")
    jitter.attrs["enabled"] = True
    jitter.attrs["type"] = "global-initial"
    jitter.attrs["insertion_point"] = "pre-encryption"
    jitter.attrs["implementation"] = "firmware-busy-loop"
    jitter.attrs["parameter_source"] = "plaintext-byte-0-low-nibble"
    jitter.attrs["label_provenance"] = "derived-exact"
    h5_file.require_group("/countermeasures/jitter/parameters")

    dataset = h5_file.create_dataset(
        spec.path,
        shape=spec.shape,
        dtype=spec.dtype,
        **chunked_dataset_options(spec, config),
    )
    digest = hashlib.sha256()
    for begin in range(0, spec.shape[0], config.io_rows):
        end = min(begin + config.io_rows, spec.shape[0])
        block = np.bitwise_and(plaintexts[begin:end, 0], np.uint8(0x0F))
        block = canonical_array(block, "u1")
        dataset[begin:end] = block
        raw = block.tobytes(order="C")
        digest.update(raw)
        progress.advance(len(raw), f"deriving {spec.path}")

    result = digest.hexdigest()
    stamp_dataset_attributes(dataset, spec, result, spec.shape[0], config)
    dataset.attrs["payload.countermeasure.jitter.loop_iterations.provenance"] = (
        "derived-exact"
    )
    return result


def infer_dataset_purpose(dataset_name: str) -> str:
    if dataset_name.endswith("_poi"):
        return "poi"
    if dataset_name.endswith("_attack"):
        return "attack"
    return "unspecified"


def infer_countermeasure_mode(key_directory: Path, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode

    lowered_parts = [part.lower() for part in key_directory.parts]
    dataset_name = key_directory.parent.name.lower()
    if "jitter" in lowered_parts or "jitter" in dataset_name:
        return "jitter-global-initial"
    if "sync" in lowered_parts or "sync" in dataset_name:
        return "none"
    raise ConversionError(
        f"{key_directory}: cannot infer whether this is a sync or jitter capture; "
        "pass --countermeasure-mode none or --countermeasure-mode jitter-global-initial"
    )


def build_metadata(
    key_directory: Path,
    arrays: SourceArrays,
    source_hashes: dict[str, str],
    config: StorageConfig,
    countermeasure_mode: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "capture.source": "ChipWhisperer",
        "capture.dataset.name": key_directory.parent.name,
        "capture.dataset.purpose": infer_dataset_purpose(key_directory.parent.name),
        "capture.dataset.key_folder": key_directory.name,
        "capture.dataset.trace_count": arrays.trace_count,
        "capture.dataset.key_generation": "chipwhisperer.ktp.Basic/python-random",
        "capture.dataset.plaintext_generation": "chipwhisperer.ktp.Basic/python-random",
        "capture.dataset.random_seed.recorded": False,
        "capture.scope.model": "ChipWhisperer-Husky",
        "capture.scope.gain.mode": "high",
        "capture.scope.gain.setting": 22,
        "capture.scope.gain.db": 25.091743119266056,
        "capture.scope.adc.multiplier": 4,
        "capture.scope.adc.decimation": 1,
        "capture.scope.adc.resolution_bits": 12,
        "capture.scope.adc.samples_per_trace": arrays.sample_count,
        "capture.scope.adc.basic_mode": "rising_edge",
        "capture.scope.adc.offset": 0,
        "capture.scope.adc.presamples": 0,
        "capture.scope.adc.segments": 1,
        "capture.scope.adc.stream_mode": False,
        "capture.sample_rate_hz": 29454545.454545453,
        "capture.clock.generator.frequency_hz": 7363636.363636363,
        "capture.clock.generator.source": "system",
        "capture.clock.generator.locked": True,
        "capture.trigger.module": "basic",
        "capture.trigger.source": "tio4",
        "capture.target.platform": "CWHUSKY",
        "capture.target.crypto.algorithm": "aes-128",
        "capture.target.crypto.implementation": "TINYAES128C",
        "capture.target.firmware.application": "simpleserial-aes",
        "capture.target.protocol.name": "SimpleSerial",
        "capture.target.protocol.version": "SS_VER_2_1",
        "capture.software.chipwhisperer.version": "6.0.0",
        "capture.software.chipwhisperer.commit": "b91e3058d83499d261caacfb54653cab9800420d",
        "origin.source.traces.file": "traces.pt",
        "origin.source.traces.sha256": source_hashes["traces"],
        "origin.source.plaintexts.file": "plain_texts.pt",
        "origin.source.plaintexts.sha256": source_hashes["plaintexts"],
        "origin.source.key.file": "key.pt",
        "origin.source.key.sha256": source_hashes["key"],
        "origin.storage.format": "hdf5",
        "origin.storage.converter": "convert_to_hdf5.py",
        "origin.storage.converter.version": CONVERTER_VERSION,
        "origin.storage.created_utc": config.created_utc,
        "origin.storage.source_preserved": True,
        "origin.storage.logical_hash.algorithm": "sha256",
        "origin.storage.chunk_target_bytes": config.chunk_target_bytes,
        "origin.storage.compression": config.compression,
        "origin.storage.compression_level": (
            config.compression_level if config.compression == "gzip" else 0
        ),
        "origin.storage.shuffle": config.shuffle,
        "origin.storage.checksum": "fletcher32" if config.fletcher32 else "none",
        "origin.storage.hdf5.h5py_version": h5py.__version__,
        "origin.storage.hdf5.library_version": h5py.version.hdf5_version,
        "origin.storage.torch.version": str(torch.__version__),
        "origin.storage.cryptography.version": str(cryptography.__version__),
    }

    if countermeasure_mode == "jitter-global-initial":
        metadata.update(
            {
                "capture.countermeasure.jitter.enabled": True,
                "capture.countermeasure.jitter.type": "global-initial",
                "capture.countermeasure.jitter.insertion_point": "pre-encryption",
                "capture.countermeasure.jitter.implementation": "firmware-busy-loop",
                "capture.countermeasure.jitter.parameter_source": "plaintext-byte-0-low-nibble",
                "capture.countermeasure.jitter.loop_iterations.min": 0,
                "capture.countermeasure.jitter.loop_iterations.max": 15,
            }
        )
    return metadata


def dataset_specs(arrays: SourceArrays, countermeasure_mode: str) -> list[DatasetSpec]:
    specs = [
        DatasetSpec(
            "/traces", tuple(arrays.traces.shape), "<f4", "trace,sample", "traces"
        ),
        DatasetSpec(
            "/plaintexts",
            tuple(arrays.plaintexts.shape),
            "u1",
            "trace,byte",
            "plaintexts",
        ),
        DatasetSpec(
            "/keys",
            tuple(arrays.key.shape),
            "u1",
            "key_byte",
            "keys",
            chunked=False,
            row_aligned=False,
        ),
        DatasetSpec(
            "/ciphertexts", (arrays.trace_count, 16), "u1", "trace,byte", "ciphertexts"
        ),
    ]
    if countermeasure_mode == "jitter-global-initial":
        specs.append(
            DatasetSpec(
                "/countermeasures/jitter/parameters/loop_iterations",
                (arrays.trace_count,),
                "u1",
                "trace",
                "countermeasures",
            )
        )
    return specs


def stamp_metadata(h5_file: Any, metadata: dict[str, Any]) -> None:
    group = h5_file.create_group("/metadata")
    for name in sorted(metadata):
        group.attrs[name] = metadata[name]


def logical_sha256_dataset(dataset: Any, spec: DatasetSpec, io_rows: int) -> str:
    digest = hashlib.sha256()
    for begin in range(0, spec.shape[0], io_rows):
        end = min(begin + io_rows, spec.shape[0])
        block = canonical_array(dataset[begin:end], spec.dtype)
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def normalized_attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def verify_output(
    path: Path,
    specs: list[DatasetSpec],
    expected_hashes: dict[str, str],
    expected_metadata: dict[str, Any],
    config: StorageConfig,
    countermeasure_mode: str,
) -> None:
    print(
        f"  {path.name}: reopening and verifying schema and logical hashes", flush=True
    )
    with h5py.File(path, "r") as h5_file:
        if normalized_attribute(h5_file.attrs.get("leakflow.schema")) != SCHEMA_NAME:
            raise ConversionError(f"{path}: invalid leakflow.schema root attribute")
        if (
            normalized_attribute(h5_file.attrs.get("leakflow.schema.version"))
            != SCHEMA_VERSION
        ):
            raise ConversionError(f"{path}: invalid schema version root attribute")

        metadata_group = h5_file.get("/metadata")
        if not isinstance(metadata_group, h5py.Group):
            raise ConversionError(f"{path}: missing /metadata group")
        misplaced_payload_metadata = [
            key for key in metadata_group.attrs if key.startswith("payload.")
        ]
        if misplaced_payload_metadata:
            raise ConversionError(
                f"{path}: role-specific payload metadata must be stored on arrays: "
                + ", ".join(sorted(misplaced_payload_metadata))
            )
        for name, expected in expected_metadata.items():
            actual = normalized_attribute(metadata_group.attrs.get(name))
            if actual != expected:
                raise ConversionError(
                    f"{path}: metadata attribute {name!r} mismatch ({actual!r} != {expected!r})"
                )

        for spec in specs:
            dataset = h5_file.get(spec.path)
            if not isinstance(dataset, h5py.Dataset):
                raise ConversionError(f"{path}: missing dataset {spec.path}")
            if tuple(dataset.shape) != spec.shape:
                raise ConversionError(
                    f"{path}:{spec.path}: shape mismatch {tuple(dataset.shape)} != {spec.shape}"
                )
            if np.dtype(dataset.dtype) != np.dtype(spec.dtype):
                raise ConversionError(
                    f"{path}:{spec.path}: dtype mismatch {dataset.dtype} != {np.dtype(spec.dtype)}"
                )
            if normalized_attribute(dataset.attrs.get("tensor.axes")) != spec.axes:
                raise ConversionError(f"{path}:{spec.path}: tensor.axes mismatch")
            if normalized_attribute(dataset.attrs.get("origin.role")) != spec.role:
                raise ConversionError(f"{path}:{spec.path}: origin.role mismatch")
            if (
                normalized_attribute(dataset.attrs.get("leakflow.row_aligned"))
                != spec.row_aligned
            ):
                raise ConversionError(
                    f"{path}:{spec.path}: leakflow.row_aligned mismatch"
                )

            expected_hash = expected_hashes[spec.path]
            stored_hash = normalized_attribute(
                dataset.attrs.get("leakflow.logical_sha256")
            )
            if stored_hash != expected_hash:
                raise ConversionError(
                    f"{path}:{spec.path}: stored logical hash mismatch"
                )
            actual_hash = logical_sha256_dataset(dataset, spec, config.io_rows)
            if actual_hash != expected_hash:
                raise ConversionError(
                    f"{path}:{spec.path}: verified logical hash mismatch"
                )

            if spec.chunked:
                expected_chunks = chunk_shape(
                    spec.shape, spec.dtype, config.chunk_target_bytes
                )
                if dataset.chunks != expected_chunks:
                    raise ConversionError(
                        f"{path}:{spec.path}: chunk shape mismatch {dataset.chunks} != {expected_chunks}"
                    )
                expected_compression = "gzip" if config.compression == "gzip" else None
                if dataset.compression != expected_compression:
                    raise ConversionError(f"{path}:{spec.path}: compression mismatch")
                if bool(dataset.shuffle) != config.shuffle:
                    raise ConversionError(
                        f"{path}:{spec.path}: shuffle setting mismatch"
                    )
                if bool(dataset.fletcher32) != config.fletcher32:
                    raise ConversionError(
                        f"{path}:{spec.path}: Fletcher32 setting mismatch"
                    )
            elif dataset.chunks is not None or dataset.compression is not None:
                raise ConversionError(
                    f"{path}:{spec.path}: fixed key must be contiguous/uncompressed"
                )

            if (
                spec.path == "/keys"
                and normalized_attribute(dataset.attrs.get("payload.crypto.key.scope"))
                != "fixed-per-file"
            ):
                raise ConversionError(
                    f"{path}:{spec.path}: fixed-key metadata mismatch"
                )
            if spec.path == "/traces":
                if (
                    normalized_attribute(dataset.attrs.get("payload.leakage.inverted"))
                    is not False
                    or "payload.leakage.polarity" in dataset.attrs
                ):
                    raise ConversionError(
                        f"{path}:{spec.path}: leakage inversion metadata mismatch"
                    )
            if spec.path == "/ciphertexts":
                if (
                    normalized_attribute(
                        dataset.attrs.get("payload.crypto.ciphertext.provenance")
                    )
                    != "derived"
                ):
                    raise ConversionError(
                        f"{path}:{spec.path}: ciphertext provenance mismatch"
                    )

        if countermeasure_mode == "none":
            if "/countermeasures" in h5_file:
                raise ConversionError(
                    f"{path}: sync capture unexpectedly has /countermeasures"
                )
            countermeasure_metadata = [
                key
                for key in metadata_group.attrs
                if key.startswith("capture.countermeasure.")
            ]
            if countermeasure_metadata:
                raise ConversionError(
                    f"{path}: sync capture unexpectedly has countermeasure metadata"
                )
        else:
            jitter = h5_file.get("/countermeasures/jitter")
            if not isinstance(jitter, h5py.Group):
                raise ConversionError(f"{path}: missing /countermeasures/jitter group")
            expected_jitter_attributes = {
                "enabled": True,
                "type": "global-initial",
                "insertion_point": "pre-encryption",
                "implementation": "firmware-busy-loop",
                "parameter_source": "plaintext-byte-0-low-nibble",
                "label_provenance": "derived-exact",
            }
            for name, expected in expected_jitter_attributes.items():
                if normalized_attribute(jitter.attrs.get(name)) != expected:
                    raise ConversionError(f"{path}: jitter attribute {name!r} mismatch")
            jitter_parameter = h5_file[
                "/countermeasures/jitter/parameters/loop_iterations"
            ]
            if (
                normalized_attribute(
                    jitter_parameter.attrs.get(
                        "payload.countermeasure.jitter.loop_iterations.provenance"
                    )
                )
                != "derived-exact"
            ):
                raise ConversionError(
                    f"{path}: jitter parameter provenance metadata mismatch"
                )


def write_hdf5(
    temporary_path: Path,
    key_directory: Path,
    arrays: SourceArrays,
    source_hashes: dict[str, str],
    config: StorageConfig,
    countermeasure_mode: str,
) -> tuple[list[DatasetSpec], dict[str, str], dict[str, Any]]:
    specs = dataset_specs(arrays, countermeasure_mode)
    specs_by_path = {spec.path: spec for spec in specs}
    metadata = build_metadata(
        key_directory, arrays, source_hashes, config, countermeasure_mode
    )
    total_bytes = sum(
        int(np.prod(spec.shape)) * np.dtype(spec.dtype).itemsize for spec in specs
    )
    progress = WriteProgress(key_directory.name, total_bytes)
    hashes: dict[str, str] = {}

    with h5py.File(temporary_path, "x", track_order=True) as h5_file:
        h5_file.attrs["leakflow.schema"] = SCHEMA_NAME
        h5_file.attrs["leakflow.schema.version"] = SCHEMA_VERSION
        stamp_metadata(h5_file, metadata)

        hashes["/traces"] = write_array_dataset(
            h5_file, specs_by_path["/traces"], arrays.traces, config, progress
        )
        hashes["/plaintexts"] = write_array_dataset(
            h5_file, specs_by_path["/plaintexts"], arrays.plaintexts, config, progress
        )
        hashes["/keys"] = write_array_dataset(
            h5_file, specs_by_path["/keys"], arrays.key, config, progress
        )
        hashes["/ciphertexts"] = write_ciphertexts(
            h5_file,
            specs_by_path["/ciphertexts"],
            arrays.plaintexts,
            arrays.key,
            config,
            progress,
        )
        jitter_path = "/countermeasures/jitter/parameters/loop_iterations"
        if countermeasure_mode == "jitter-global-initial":
            hashes[jitter_path] = write_jitter_iterations(
                h5_file,
                specs_by_path[jitter_path],
                arrays.plaintexts,
                config,
                progress,
            )
        h5_file.flush()

    with temporary_path.open("rb") as output:
        os.fsync(output.fileno())
    return specs, hashes, metadata


def source_file_hashes(key_directory: Path) -> dict[str, str]:
    print(f"  {key_directory.name}: hashing source files", flush=True)
    return {
        "traces": sha256_file(key_directory / "traces.pt"),
        "plaintexts": sha256_file(key_directory / "plain_texts.pt"),
        "key": sha256_file(key_directory / "key.pt"),
    }


def convert_key_directory(
    key_directory: Path,
    config: StorageConfig,
    requested_countermeasure_mode: str,
    overwrite: bool,
) -> str:
    destination = key_directory.with_suffix(".h5")
    if destination.exists() and not overwrite:
        print(f"skip: {destination} already exists (use --overwrite to replace it)")
        return "skipped"
    if destination.exists() and not destination.is_file():
        raise ConversionError(
            f"{destination}: destination exists and is not a regular file"
        )

    countermeasure_mode = infer_countermeasure_mode(
        key_directory, requested_countermeasure_mode
    )
    arrays = load_and_validate_source(key_directory)
    source_hashes = source_file_hashes(key_directory)
    temporary_path = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )

    try:
        specs, hashes, metadata = write_hdf5(
            temporary_path,
            key_directory,
            arrays,
            source_hashes,
            config,
            countermeasure_mode,
        )
        verify_output(
            temporary_path,
            specs,
            hashes,
            metadata,
            config,
            countermeasure_mode,
        )
        os.replace(temporary_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(
        f"converted: {key_directory} -> {destination} "
        f"(source preserved, countermeasures={countermeasure_mode})",
        flush=True,
    )
    return "converted"


def is_key_directory(path: Path) -> bool:
    return path.is_dir() and KEY_DIRECTORY_PATTERN.fullmatch(path.name) is not None


def discover_key_directories(roots: Iterable[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for root in roots:
        if not root.exists():
            raise ConversionError(f"Input path does not exist: {root}")
        if not root.is_dir():
            raise ConversionError(f"Input path is not a directory: {root}")
        if is_key_directory(root):
            discovered.add(root.resolve())
            continue
        for candidate in root.rglob("key_*"):
            if is_key_directory(candidate):
                discovered.add(candidate.resolve())

    if not discovered:
        rendered = ", ".join(str(root) for root in roots)
        raise ConversionError(f"No key_NN source folders found under: {rendered}")
    return sorted(discovered, key=lambda path: str(path))


def resolve_created_utc(explicit: str | None) -> str:
    if explicit:
        try:
            parsed = dt.datetime.fromisoformat(explicit.replace("Z", "+00:00"))
        except ValueError as error:
            raise ConversionError(
                f"Invalid --created-utc timestamp: {explicit}"
            ) from error
        if parsed.tzinfo is None:
            raise ConversionError(
                "--created-utc must include a UTC offset or trailing Z"
            )
        return (
            parsed.astimezone(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch is not None:
        try:
            timestamp = int(source_date_epoch)
        except ValueError as error:
            raise ConversionError(
                "SOURCE_DATE_EPOCH must be an integer Unix timestamp"
            ) from error
        return (
            dt.datetime.fromtimestamp(timestamp, dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert key_NN Torch tensor folders to verified sibling key_NN.h5 files. "
            "Source folders are always preserved."
        )
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="One or more key_NN folders or trees containing key_NN folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing key_NN.h5 files after verification.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered source and destination paths without loading or writing data.",
    )
    parser.add_argument(
        "--countermeasure-mode",
        choices=("auto", "none", "jitter-global-initial"),
        default="auto",
        help=(
            "Countermeasure schema to write. auto recognizes sync/jitter path names; "
            "default: auto."
        ),
    )
    parser.add_argument(
        "--compression",
        choices=("none", "gzip"),
        default="gzip",
        help="Compression for non-tiny arrays; default: gzip.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=1,
        help="Gzip compression level from 0 to 9; default: 1.",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable byte shuffle. Default: enabled with gzip, disabled with none.",
    )
    parser.add_argument(
        "--fletcher32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-chunk Fletcher32 checksums; default: enabled.",
    )
    parser.add_argument(
        "--chunk-target-bytes",
        type=int,
        default=DEFAULT_CHUNK_TARGET_BYTES,
        help=f"Logical target bytes per chunk; default: {DEFAULT_CHUNK_TARGET_BYTES}.",
    )
    parser.add_argument(
        "--io-rows",
        type=int,
        default=DEFAULT_IO_ROWS,
        help=f"Rows processed per converter I/O operation; default: {DEFAULT_IO_ROWS}.",
    )
    parser.add_argument(
        "--created-utc",
        help=(
            "Explicit ISO-8601 conversion timestamp. SOURCE_DATE_EPOCH is honored when omitted."
        ),
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 <= args.compression_level <= 9:
        raise ConversionError("--compression-level must be between 0 and 9")
    if args.chunk_target_bytes <= 0:
        raise ConversionError("--chunk-target-bytes must be greater than zero")
    if args.io_rows <= 0:
        raise ConversionError("--io-rows must be greater than zero")

    roots = [path.expanduser().resolve() for path in args.roots]
    key_directories = discover_key_directories(roots)
    print(f"discovered {len(key_directories)} key folder(s)", flush=True)
    if args.dry_run:
        for key_directory in key_directories:
            print(f"  {key_directory} -> {key_directory.with_suffix('.h5')}")
        return 0

    load_dependencies()
    shuffle = args.compression == "gzip" if args.shuffle is None else args.shuffle
    config = StorageConfig(
        compression=args.compression,
        compression_level=args.compression_level,
        shuffle=shuffle,
        fletcher32=args.fletcher32,
        chunk_target_bytes=args.chunk_target_bytes,
        io_rows=args.io_rows,
        created_utc=resolve_created_utc(args.created_utc),
    )

    converted = 0
    skipped = 0
    for index, key_directory in enumerate(key_directories, start=1):
        print(f"[{index}/{len(key_directories)}] {key_directory}", flush=True)
        status = convert_key_directory(
            key_directory,
            config,
            args.countermeasure_mode,
            args.overwrite,
        )
        converted += status == "converted"
        skipped += status == "skipped"

    print(
        f"complete: converted={converted}, skipped={skipped}, "
        f"source_folders_preserved={len(key_directories)}",
        flush=True,
    )
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except ConversionError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
