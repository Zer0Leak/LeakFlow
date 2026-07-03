# Crypto Module Context

Use this for work touching generic crypto/SCA leakage helpers.

## Files

Public headers:

- `include/leakflow/crypto`

Sources:

- `src/crypto`

Tests:

- `tests/crypto`

## Target

- `leakflow_crypto`

## Dependencies

`leakflow_crypto` depends on:

- `leakflow_base`
- LibTorch through `leakflow_base`

`leakflow_core` must not depend on `leakflow_crypto`.

## Current API

- Scalar Hamming weight and Hamming distance helpers for 1, 2, 4, and 8 byte
  unsigned values.
- Torch `uint8` Hamming weight and Hamming distance helpers.
- AES S-box helper.
- AES first-round S-box leakage helpers for scalar bytes and Torch tensors.
- AES first-round leakage-channel helpers for known-key leakage `[U,N,L]` and
  guess-domain hypotheses `[U,G,N,L]`.

## AES First-Round S-Box Leakage Contract

For key byte `k` and plaintext byte `m`:

```text
y = AES_SBOX[m XOR k]
```

The public leakage outputs are:

- `m`,
- `m XOR k`,
- `y`,
- `HW(m)`,
- `HW(m_xor_k)`,
- `HW(y)`.

Torch leakage helpers return:

- `values`: `[N,2]` `uint8`, columns `m`, `y`,
- `hamming_weights`: `[N,2]` `uint8`, columns `HW(m)`, `HW(y)`.

Byte-index-list Torch leakage helpers return:

- `values`: `[B,N,2]` `uint8`, where axis 0 follows the requested byte-index
  order,
- `hamming_weights`: `[B,N,2]` `uint8`, where axis 0 follows the requested
  byte-index order.

The shared first-round leakage-channel helper accepts ordered channel values
`HW(m)`, `HW(m_xor_k)`, `HW(y)`, and the S-box output bit channels `y(0)` through
`y(7)` (`y(0)` is the least significant bit). Known-key mode returns:

- `leakage`: `[U,N,L]` `uint8`,

where `U` follows the requested byte-index order and `L` follows the requested
channel order.

Guess-domain mode accepts `guess_values [G]` and returns:

- `hypotheses`: `[U,G,N,L]` `uint8`.

`HW(m)` is guess/key-independent. `HW(m_xor_k)`, `HW(y)`, and `y(0)` through
`y(7)` are guess/key-dependent.

Accepted Torch key inputs:

- scalar `uint8`,
- `[1]` scalar-like `uint8`,
- `[N]` per-trace key bytes,
- `[N,16]` full key blocks for byte-index APIs.

Accepted plaintext inputs:

- `[N]` byte vectors,
- `[N,16]` plaintext blocks for byte-index APIs.

## Out Of Scope Unless Requested

- Pipeline elements.
- `leakflow_plugins_crypto`.
- CPA/Pearson/PoI search.
- AES full encryption/decryption.
- AES key schedule.
- Kyber / ML-KEM.
- Custom CUDA kernels.
