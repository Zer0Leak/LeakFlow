# LeakFlow SCA Tensor Dataset Schema

Schema name: `leakflow.sca.tensor-dataset`

Schema version: `1`

This is a storage-neutral schema for LeakFlow side-channel tensor datasets. The
same paths, shapes, logical dtypes, axes, metadata names, and logical hashes are
intended for HDF5 and Zarr. Storage-specific chunk and codec settings are
recorded separately so the two formats can later be compared without changing
the logical dataset.

## Root attributes

Every store has these root attributes:

| Attribute | Value |
|---|---|
| `leakflow.schema` | `leakflow.sca.tensor-dataset` |
| `leakflow.schema.version` | `1` |

Readers must reject an unsupported schema name or version.

## Required arrays

Each `key_NN` store represents traces captured with one fixed AES-128 key.

| Path | Logical dtype and shape | `tensor.axes` | Meaning |
|---|---|---|---|
| `/traces` | `float32 [N,5000]` | `trace,sample` | Sign-inverted power traces |
| `/plaintexts` | `uint8 [N,16]` | `trace,byte` | AES input blocks |
| `/keys` | `uint8 [16]` | `key_byte` | Fixed AES key for this store |
| `/ciphertexts` | `uint8 [N,16]` | `trace,byte` | AES-128 output derived from plaintext and key |

`N` must be positive and equal for every trace-indexed array. `/keys` is not
repeated per trace. Ciphertexts are software-derived because target `textout`
was not captured. The `/ciphertexts` array explicitly records this fact.

Every array has these format-neutral attributes:

- `tensor.axes`: comma-separated semantic axis names from the table above.
- `origin.role`: semantic role such as `traces`, `plaintexts`, `keys`,
  `ciphertexts`, or `countermeasures`.
- `leakflow.logical_sha256`: SHA-256 of the canonical logical bytes.
- `leakflow.logical_nbytes`: uncompressed logical byte count.
- `leakflow.row_aligned`: `true` when axis 0 indexes capture rows and `false`
  for the fixed `/keys` array.

Readers use `leakflow.row_aligned`, rather than tensor rank alone, to decide
whether to slice an array by a requested trace range. It is `true` for
`/traces`, `/plaintexts`, `/ciphertexts`, and all current trace-indexed
countermeasure arrays. It is `false` for `/keys`, which must be emitted unchanged
for every selected trace range.

Canonical logical bytes are the array in C row-major order, using little-endian
IEEE-754 `float32` for traces and endian-independent `uint8` for byte arrays.
These hashes must match across equivalent HDF5 and Zarr conversions even when
their physical encodings differ.

## Countermeasures

A synchronized capture has no `/countermeasures` group and no
`capture.countermeasure.*` metadata. Absence means that no countermeasure is
present; synchronized files must not manufacture a zero-valued label array.

The current jitter capture adds:

| Path | Logical dtype and shape | `tensor.axes` | Meaning |
|---|---|---|---|
| `/countermeasures/jitter/parameters/loop_iterations` | `uint8 [N]` | `trace` | Firmware busy-loop count before AES |

The value is an exact derived label:

```text
loop_iterations = plaintexts[:, 0] & 0x0f
```

The `/countermeasures/jitter` group has these attributes:

| Attribute | Value |
|---|---|
| `enabled` | `true` |
| `type` | `global-initial` |
| `insertion_point` | `pre-encryption` |
| `implementation` | `firmware-busy-loop` |
| `parameter_source` | `plaintext-byte-0-low-nibble` |
| `label_provenance` | `derived-exact` |

The hierarchy deliberately separates a countermeasure's parameters from future
annotations. Later datasets can add arrays such as
`/countermeasures/jitter/annotations/region_samples` without changing the
meaning of `parameters/loop_iterations`. Other countermeasure families can add
their own sibling groups.

## Metadata

`/metadata` is a group whose attributes follow LeakFlow's metadata taxonomy:

- `capture.*` records durable acquisition and dataset facts.
- `origin.*` records source-file provenance.
- `origin.storage.*` records physical encoding and converter provenance.

Role-specific `payload.*` attributes live on the array whose values they
describe, so plaintexts and keys never inherit trace-leakage facts.

The AES converter records, among other fields:

```text
capture.source=ChipWhisperer
capture.scope.model=ChipWhisperer-Husky
capture.target.platform=CWHUSKY
capture.target.crypto.algorithm=aes-128
capture.target.crypto.implementation=TINYAES128C
capture.scope.gain.setting=22
capture.scope.gain.db=25.091743119266056
capture.clock.generator.frequency_hz=7363636.363636363
capture.sample_rate_hz=29454545.454545453
```

Jitter stores additionally record canonical `capture.countermeasure.jitter.*`
facts. Synchronized stores omit those fields.

Array-specific payload facts stay on the array they describe instead of being
copied into common `/metadata`:

```text
/traces:      payload.leakage.inverted=false
/keys:        payload.crypto.key.scope=fixed-per-file
/ciphertexts: payload.crypto.ciphertext.provenance=derived
/ciphertexts: payload.crypto.ciphertext.derivation=aes-128-encrypt(plaintext,key)
/countermeasures/jitter/parameters/loop_iterations:
               payload.countermeasure.jitter.loop_iterations.provenance=derived-exact
```

The converter also records capture settings from the dataset readme, data
generation facts, SHA-256 hashes of the three source `.pt` files, converter and
dependency versions, storage settings, and an ISO-8601 UTC conversion timestamp.
`SOURCE_DATE_EPOCH` or `--created-utc` can make the timestamp deterministic.

## HDF5 baseline encoding

The portable default for non-tiny arrays is:

```text
compression=gzip
compression_level=1
shuffle=true
checksum=fletcher32
chunk_target_bytes=1048576
```

Chunks span complete rows and contain at most the requested logical target size
(unless one row itself is larger). `/keys` is only 16 bytes and remains
contiguous and uncompressed. Each array records its actual physical settings in
`origin.storage.*` attributes. The converter supports `--compression none`; by
default that also disables shuffle while retaining Fletcher32, and explicit
`--shuffle`/`--no-fletcher32` flags allow controlled benchmarks.

Fair HDF5/Zarr comparisons should distinguish:

1. Identical logical arrays and chunk shapes with no compression.
2. Identical logical arrays and chunk shapes with gzip level 1.
3. Separately labeled format-native optimized configurations.

Report logical bytes, logical SHA-256 values, physical file/store bytes, read
batch size, cold/warm cache conditions, and elapsed read/decompression time.

## Conversion

From the repository root, preview every AES key folder:

```bash
python3 traces/aes/convert_to_hdf5.py --dry-run traces/aes/sync traces/aes/jitter
```

Convert the complete tree with the portable defaults:

```bash
python3 traces/aes/convert_to_hdf5.py traces/aes/sync traces/aes/jitter
```

Convert a single folder without compression:

```bash
python3 traces/aes/convert_to_hdf5.py --compression none \
  traces/aes/sync/aes_sync_poi/key_01
```

The required Python packages are `numpy`, `torch`, `h5py`, and `cryptography`.
For each folder the converter writes a uniquely named temporary sibling,
reopens it, validates schema, shapes, dtypes, filters, metadata, and every
logical hash, then atomically renames it to `key_NN.h5`. A failed conversion
leaves an existing destination untouched. Source folders are always preserved;
the converter has no deletion option. Use `--overwrite` only to atomically
replace an existing verified HDF5 file.
