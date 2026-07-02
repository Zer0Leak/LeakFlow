# AES Synchronized Trace Dataset

Documented on: 2026-05-31
Author: Zer0Leak

## ChipWhisperer And Firmware

- ChipWhisperer Python package version in this checkout: `6.0.0`
- ChipWhisperer source checkout commit: `b91e3058d83499d261caacfb54653cab9800420d`
- Jupyter submodule checkout commit: `fe7e2aa1585798c3a5a6b247f231abe2e2b4a2cd`
- Scope/platform setting in the notebook: `SCOPETYPE = 'OPENADC'`, `PLATFORM = 'CWHUSKY'`
- Target firmware application: `firmware/mcu/simpleserial-aes/simpleserial-aes.c`
- Programmed firmware image: `firmware/mcu/simpleserial-aes/simpleserial-aes-CWHUSKY.hex`
- Crypto target: `TINYAES128C`
- AES implementation selected by build system: `tiny-AES128-C`, compiled with `-DTINYAES128C`
- SimpleSerial firmware/protocol version: `SS_VER_2_1`
- `SS_VER_2_1` value in `firmware/mcu/simpleserial/simpleserial.h`: `3`

Connected scope:

- Detected scope hardware: `ChipWhisperer-Husky`
- Scope serial number: `5xxxx1`
- Scope USB/control firmware version: `1.7.0`
- Scope FPGA bitstream build time: `5/26/2026, 09:27`
- Scope hardware register version tuple: `[2, 10, "ChipWhisperer-Husky", 0]`
- ADC multiplier: `4`
- ADC sampling frequency: `29,454,545.454545453 samples/s` (`29.454545 MHz`)
- ADC decimation/downsampling: `1`
- Effective saved-trace sample rate: `29,454,545.454545453 samples/s` (`29.454545 MS/s`)
- Clock generator frequency: `7,363,636.363636363 Hz` (`7.363636 MHz`)
- Clock generator source: `system`
- Clock generator locked: `true`
- External clock error flag: `false`

Capture-critical scope settings:

- Gain mode: `high`
- Gain: `25.091743119266056 dB`
- ADC samples per trace: `5000`
- ADC bits per sample: `12`
- ADC basic mode: `rising_edge`
- ADC offset: `0`
- ADC presamples: `0`
- ADC segments: `1`
- ADC stream mode: `false`
- ADC clip errors disabled: `false`
- Trigger module: `basic`
- Trigger source: `tio4`
- Target clock output: `scope.io.hs2 = "clkgen"`
- Target power enabled: `true`
- I/O routing: `tio1 = "serial_rx"`, `tio2 = "serial_tx"`, `tio3 = "high_z"`, `tio4 = "high_z"`

The target AES firmware/protocol version used by the programmed target remains the `SS_VER_2_1` setting above.

## Build And Programming Commands

The notebook used these settings:

```python
SCOPETYPE = "OPENADC"
PLATFORM = "CWHUSKY"
CRYPTO_TARGET = "TINYAES128C"
SS_VER = "SS_VER_2_1"
```

The notebook build cell was:

```bash
%%bash -s "$PLATFORM" "$CRYPTO_TARGET" "$SS_VER"
cd ../../../firmware/mcu/simpleserial-aes
make PLATFORM=$1 CRYPTO_TARGET=$2 SS_VER=$3 -j
```

The notebook programming cell was:

```python
cw.program_target(scope, prog, "../../../firmware/mcu/simpleserial-aes/simpleserial-aes-{}.hex".format(PLATFORM))
```

## Dataset Layout

Two datasets are generated beside the notebook:

```text
aes_sync_poi/
  key_01/
    key.pt
    plain_texts.pt
    traces.pt
  ...
  key_50/
    key.pt
    plain_texts.pt
    traces.pt

aes_sync_attack/
  key_01/
    key.pt
    plain_texts.pt
    traces.pt
  ...
  key_30/
    key.pt
    plain_texts.pt
    traces.pt
```

Dataset sizes:

- `aes_sync_poi`: 50 independent AES keys, 2,000 traces per key, 100,000 traces total.
- `aes_sync_attack`: 30 independent AES keys, 10,000 traces per key, 300,000 traces total.

Tensor formats:

- `key.pt`: Torch `uint8`, shape `(16,)`.
- `plain_texts.pt`: Torch `uint8`, shape `(traces_per_key, 16)`.
- `traces.pt`: Torch `float32`, shape `(traces_per_key, num_samples)`.
- In this capture configuration, completed trace files were observed with `num_samples = 5000`.

Each `key_XX` folder contains one fixed AES key. Every trace in that folder was captured with that same key and a newly generated plaintext.

## Randomness Of Keys And Plaintexts

The capture code uses:

```python
ktp = cw.ktp.Basic()
ktp.fixed_key = False
ktp.fixed_text = False
```

In ChipWhisperer's `cw.ktp.Basic()` acquisition pattern, setting `fixed_key = False` makes each key byte generated with Python's `random.randint(0, 255)`. Setting `fixed_text = False` does the same for every plaintext byte.

So:

- Keys are intended to be uniformly random over 16 independent bytes.
- Plaintexts are intended to be uniformly random over 16 independent bytes.
- One key is generated per `key_XX` folder and then held fixed for that folder.
- One plaintext is generated per captured trace.
- No explicit random seed is set in the capture cell.
- The randomness comes from Python's standard `random` module through ChipWhisperer's KTP helper; it is suitable for randomized acquisition, but should not be described as cryptographically secure randomness.

## Inverted Signal Convention

ChipWhisperer power measurements are often negative-going for higher target power consumption because the ADC observes voltage around the shunt/measurement path. In this dataset the saved traces are sign-flipped:

```python
wave = np.asarray(trace.wave, dtype=np.float32)
if INVERT_TRACES:
    wave = -wave
```

This means the saved `traces.pt` files contain `-trace.wave`, not the raw ChipWhisperer waveform.

Important details:

- The inversion is a sign flip, not an absolute value operation.
- Negative-going raw power events become positive-going in the saved dataset.
- Depending on ADC offset and noise, individual saved samples may still be negative.
- The intended convention is that higher relative saved trace values correspond to higher target power consumption.

## Python Capture Source

This is the Python source used in the notebook cell that generated the datasets:

```python
from pathlib import Path
from tqdm.notebook import tqdm, trange
import numpy as np
import time

ktp = cw.ktp.Basic()
ktp.fixed_key = False
ktp.fixed_text = False

POI_NUM_KEYS = 50
POI_TRACES_PER_KEY = 2000

ATTACK_NUM_KEYS = 30
ATTACK_TRACES_PER_KEY = 10000

KEY_LEN = 16
TEXT_LEN = 16
INVERT_TRACES = True
RETRY_SLEEP_S = 0.05
MAX_CONSECUTIVE_TIMEOUTS = 20


def _next_key():
    key = bytearray(ktp.next_key())
    if len(key) != KEY_LEN:
        raise ValueError(f"Expected {KEY_LEN}-byte AES key, got {len(key)} bytes")
    return key


def _next_text():
    text = bytearray(ktp.next_text())
    if len(text) != TEXT_LEN:
        raise ValueError(f"Expected {TEXT_LEN}-byte AES plaintext, got {len(text)} bytes")
    return text


def capture_dataset(output_dir, num_keys, traces_per_key):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key_index in trange(num_keys, desc=f"{output_dir.name}: keys"):
        key_dir = output_dir / f"key_{key_index + 1:02d}"
        key_dir.mkdir(parents=True, exist_ok=True)

        key = _next_key()
        key_array = np.frombuffer(bytes(key), dtype=np.uint8).copy()
        plain_texts = np.empty((traces_per_key, TEXT_LEN), dtype=np.uint8)
        traces = None
        captured = 0
        retries = 0
        consecutive_timeouts = 0

        with tqdm(total=traces_per_key, desc=key_dir.name, leave=False) as progress:
            while captured < traces_per_key:
                text = _next_text()
                trace = cw.capture_trace(scope, target, text, key)

                if trace is None:
                    retries += 1
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                        raise RuntimeError(f"Too many consecutive capture timeouts for {key_dir}")
                    time.sleep(RETRY_SLEEP_S)
                    continue

                consecutive_timeouts = 0

                wave = np.asarray(trace.wave, dtype=np.float32)
                if INVERT_TRACES:
                    wave = -wave

                if traces is None:
                    traces = np.empty((traces_per_key, wave.size), dtype=np.float32)

                plain_texts[captured] = np.frombuffer(bytes(text), dtype=np.uint8)
                traces[captured] = wave
                captured += 1
                progress.update(1)

        np.save(key_dir / "key.pt", key_array)
        np.save(key_dir / "plain_texts.pt", plain_texts)
        np.save(key_dir / "traces.pt", traces)

        if retries:
            print(f"{key_dir}: retried {retries} timed-out capture(s)")

    return output_dir


poi_dir = capture_dataset("aes_sync_poi", POI_NUM_KEYS, POI_TRACES_PER_KEY)
attack_dir = capture_dataset("aes_sync_attack", ATTACK_NUM_KEYS, ATTACK_TRACES_PER_KEY)

trace_array = np.load(poi_dir / "key_01" / "traces.pt", mmap_mode="r")
textin_array = np.load(poi_dir / "key_01" / "plain_texts.pt", mmap_mode="r")
```

THEN CONVERTED TO PT (TORCH LATER)

## Target Firmware Source

This is the main target firmware source copied from:

`firmware/mcu/simpleserial-aes/simpleserial-aes.c`

```c
/*
    This file is part of the ChipWhisperer Example Targets
    Copyright (C) 2012-2017 NewAE Technology Inc.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "aes-independant.h"
#include "hal.h"
#include "simpleserial.h"
#include <stdint.h>
#include <stdlib.h>

uint8_t get_mask(uint8_t* m, uint8_t len)
{
  aes_indep_mask(m, len);
  return 0x00;
}

uint8_t get_key(uint8_t* k, uint8_t len)
{
	aes_indep_key(k);
	return 0x00;
}

uint8_t get_pt(uint8_t* pt, uint8_t len)
{
    aes_indep_enc_pretrigger(pt);

	trigger_high();

  #ifdef ADD_JITTER
  for (volatile uint8_t k = 0; k < (*pt & 0x0F); k++);
  #endif

	aes_indep_enc(pt); /* encrypting the data block */
	trigger_low();

    aes_indep_enc_posttrigger(pt);

	simpleserial_put('r', 16, pt);
	return 0x00;
}

uint8_t reset(uint8_t* x, uint8_t len)
{
    // Reset key here if needed
	return 0x00;
}

static uint16_t num_encryption_rounds = 10;

uint8_t enc_multi_getpt(uint8_t* pt, uint8_t len)
{
    aes_indep_enc_pretrigger(pt);

    for(unsigned int i = 0; i < num_encryption_rounds; i++){
        trigger_high();
        aes_indep_enc(pt);
        trigger_low();
    }

    aes_indep_enc_posttrigger(pt);
	simpleserial_put('r', 16, pt);
    return 0;
}

uint8_t enc_multi_setnum(uint8_t* t, uint8_t len)
{
    //Assumes user entered a number like [0, 200] to mean "200"
    //which is most sane looking for humans I think
    num_encryption_rounds = t[1];
    num_encryption_rounds |= t[0] << 8;
    return 0;
}

#if SS_VER == SS_VER_2_1
uint8_t aes(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    uint8_t req_len = 0;
    uint8_t err = 0;
    uint8_t mask_len = 0;
    if (scmd & 0x04) {
        // Mask has variable length. First byte encodes the length
        mask_len = buf[req_len];
        req_len += 1 + mask_len;
        if (req_len > len) {
            return SS_ERR_LEN;
        }
        err = get_mask(buf + req_len - mask_len, mask_len);
        if (err)
            return err;
    }

    if (scmd & 0x02) {
        req_len += 16;
        if (req_len > len) {
            return SS_ERR_LEN;
        }
        err = get_key(buf + req_len - 16, 16);
        if (err)
            return err;
    }
    if (scmd & 0x01) {
        req_len += 16;
        if (req_len > len) {
            return SS_ERR_LEN;
        }
        err = get_pt(buf + req_len - 16, 16);
        if (err)
            return err;
    }

    if (len != req_len) {
        return SS_ERR_LEN;
    }

    return 0x00;

}
#endif

int main(void)
{
	uint8_t tmp[KEY_LENGTH] = {DEFAULT_KEY};

    platform_init();
    init_uart();
    trigger_setup();

	aes_indep_init();
	aes_indep_key(tmp);

    /* Uncomment this to get a HELLO message for debug */

    // putch('h');
    // putch('e');
    // putch('l');
    // putch('l');
    // putch('o');
    // putch('\n');

	simpleserial_init();
    #if SS_VER == SS_VER_2_1
    simpleserial_addcmd(0x01, 16, aes);
    #else
    simpleserial_addcmd('k', 16, get_key);
    simpleserial_addcmd('p', 16,  get_pt);
    simpleserial_addcmd('x',  0,   reset);
    simpleserial_addcmd_flags('m', 18, get_mask, CMD_FLAG_LEN);
    simpleserial_addcmd('s', 2, enc_multi_setnum);
    simpleserial_addcmd('f', 16, enc_multi_getpt);
    #endif
    while(1)
        simpleserial_get();
}
```
