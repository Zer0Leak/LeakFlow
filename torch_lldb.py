# torch_lldb.py
# LLDB Python module for displaying PyTorch tensors in Watch/Variables.
# Requires a C helper linked into the debug target:
#   extern "C" const char* dtv(const torch::Tensor*);

import lldb


def _expr_path(valobj: lldb.SBValue) -> str:
    s = lldb.SBStream()
    valobj.GetExpressionPath(s)
    p = s.GetData()
    return p if p else (valobj.GetName() or "")


def _unquote(s: str) -> str:
    if not s:
        return ""
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s.replace(r"\"", '"')


def tensor_summary(valobj: lldb.SBValue, _internal_dict) -> str:
    # CodeLLDB's Watch evaluator may expose a C++ reference through an
    # expression path such as `*(targets)`, even though `targets` is already a
    # reference. Calling dtv through the value's load address avoids generating
    # invalid expressions such as `&(*(targets))`.
    address = valobj.GetLoadAddress()
    if address != lldb.LLDB_INVALID_ADDRESS:
        expr = f"dtv((const at::Tensor*){address})"
    else:
        p = _expr_path(valobj)
        if not p:
            return ""
        expr = f"dtv(&({p}))"
    opts = lldb.SBExpressionOptions()
    opts.SetTimeoutInMicroSeconds(200000)  # 200 ms
    opts.SetFetchDynamicValue(lldb.eDynamicCanRunTarget)
    opts.SetTryAllThreads(False)

    v = valobj.GetFrame().EvaluateExpression(expr, opts)
    if not v.IsValid() or v.GetError().Fail():
        error = v.GetError().GetCString() if v.IsValid() else "invalid result"
        return f"<tensor: eval failed: {error}>"

    return _unquote(v.GetSummary() or v.GetValue() or "")


def __lldb_init_module(debugger: lldb.SBDebugger, _internal_dict):
    # Register summaries; do NOT clear existing ones.
    # Explicit names (your debug shows (at::Tensor)).
    debugger.HandleCommand('type summary add -F torch_lldb.tensor_summary "at::Tensor"')
    debugger.HandleCommand('type summary add -F torch_lldb.tensor_summary "at::Tensor &"')
    debugger.HandleCommand('type summary add -F torch_lldb.tensor_summary "at::Tensor const &"')

    # Harmless extras in case some compilation units expose torch::Tensor.
    debugger.HandleCommand('type summary add -F torch_lldb.tensor_summary "torch::Tensor"')
    debugger.HandleCommand('type summary add -F torch_lldb.tensor_summary "torch::Tensor &"')
    debugger.HandleCommand('type summary add -F torch_lldb.tensor_summary "torch::Tensor const &"')

    # Regex catch-all for slight spelling variations.
    debugger.HandleCommand('type summary add -x -F torch_lldb.tensor_summary "^(at|torch)::Tensor( const)?( &)?$"')

    debugger.HandleCommand('script print(">>> torch_lldb.py loaded: tensor summaries registered <<<")')
