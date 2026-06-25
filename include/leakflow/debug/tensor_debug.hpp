#pragma once

// Tensor debugging helpers (the `leakflow_torch_debug` module).
//
// Small, CPU-safe string renderers for torch::Tensor plus C-linkage entry points
// (pt/ptv/dtv/ps) that an LLDB/GDB session can call to inspect tensors in the
// Variables/Watch view or the console. See torch_lldb.py and
// .vscode/launch.json for how the debugger loads and uses these.
//
// This is a reusable, debug-only utility. It is built as an OBJECT library so its
// symbols link directly into whatever executable pulls it in (never pruned from a
// static archive), and is gated by the CMake option LEAKFLOW_TORCH_DEBUG
// (default ON). It depends only on <torch/torch.h> — no other LeakFlow internals,
// no xtensor, no training code.

#include <torch/torch.h>

#include <string>

namespace leakflow::debug {

// Portable scalar-formatting settings (debug-only).
//
// Not thread-safe: the global instance below is intended to be read/written from
// a single thread (the one a debugger has stopped). Do not mutate it from the
// live executor's segment threads.
struct FormatSettings {
    bool scientific = true;   // false => std::fixed
    int precision = 6;        // std::setprecision
    bool right_align = true;  // false => std::left
    int width = 0;            // 0 => no explicit std::setw
};

extern FormatSettings g_default_format_settings;

// RAII override of the global formatting settings.
struct FormatGuard {
    FormatSettings old_settings;

    explicit FormatGuard(FormatSettings new_settings)
        : old_settings(g_default_format_settings) {
        g_default_format_settings = new_settings;
    }
    ~FormatGuard() { g_default_format_settings = old_settings; }

    FormatGuard(const FormatGuard&) = delete;
    FormatGuard& operator=(const FormatGuard&) = delete;
};

// 0-dim (scalar) tensor -> string. Throws if the tensor is undefined or non-scalar.
std::string tsstr(const torch::Tensor& s);

// Tensor -> "(shape=..., dtype=..., dev=..., req_grad=...) <values>".
// CPU-safe: detaches and moves CUDA tensors to CPU before reading values.
//   indent=true  -> pretty, multi-line
//   indent=false -> single line (used by the dtv summary)
std::string tstr(const torch::Tensor& t, bool indent = true);

}  // namespace leakflow::debug

// Debugger entry points (C linkage so they are easy to call from an LLDB/GDB
// expression, e.g. `p (char*)dtv(&some_tensor)`). Each returns a pointer to a
// thread-local buffer valid until the next call on the same thread.
extern "C" {
const char* pt(const torch::Tensor* t);   // header only: shape/dtype/device/grad
const char* ptv(const torch::Tensor* t);  // indented values
const char* dtv(const torch::Tensor* t);  // single-line values (used by summaries)
const char* ps(const torch::Tensor* t);   // shape only
}
