#include "leakflow/debug/tensor_debug.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>

namespace leakflow::debug {

FormatSettings g_default_format_settings;

namespace {

std::string tensor_header_str(const torch::Tensor& t) {
    std::ostringstream oss;
    oss << "Tensor(sizes=" << t.sizes() << ", dtype=" << t.dtype() << ", device=" << t.device()
        << ", requires_grad=" << (t.requires_grad() ? "true" : "false") << ")";
    return oss.str();
}

std::string sizes_str(c10::IntArrayRef s) {
    std::ostringstream oss;
    oss << s;
    return oss.str();
}

void apply_format(std::ostream& os) {
    const auto& f = g_default_format_settings;
    os << (f.scientific ? std::scientific : std::fixed) << std::setprecision(f.precision)
       << (f.right_align ? std::right : std::left);
    if (f.width > 0) {
        os << std::setw(f.width);
    }
}

// Render tensor values as nested, comma-separated brackets. Assumes a detached
// CPU tensor (tstr guarantees this before calling).
std::string render_tensor_values_compact(const torch::Tensor& x, const int64_t max_scalar_to_show,
                                          const int64_t max_scalars_to_show_per_1D_vector, bool indent) {
    constexpr int indent_size = 4;
    if (x.dim() == 0) {
        return "(" + tsstr(x) + ")";
    }

    int64_t scalars_shown = 0;
    bool scalar_dropped = false;

    std::function<std::string(const torch::Tensor&, int)> render =
        [&](const torch::Tensor& t, int level) -> std::string {
        const int64_t ndim = t.dim();
        if (ndim == 0) {
            ++scalars_shown;
            return tsstr(t);
        }

        auto indentation = std::string(static_cast<std::size_t>(level) * indent_size, ' ');
        int64_t n = t.sizes()[0];
        if (ndim == 1) {
            n = std::min(t.sizes()[0], max_scalars_to_show_per_1D_vector);
        }
        const bool print_etc = (t.sizes()[0] > n);

        std::string out;
        if (indent) {
            out += indentation + "[";
            if (ndim > 1) {
                out += "\n";
            }
        } else {
            out += "[";
        }

        for (int64_t i = 0; i < n; ++i) {
            if (scalars_shown >= max_scalar_to_show) {
                scalar_dropped = true;
                break;
            }
            if (i > 0) {
                out.push_back(',');
                if (indent) {
                    out += (ndim > 1) ? "\n" : " ";
                }
            }
            out += render(t.select(0, i), level + 1);
        }

        if (ndim == 1 && print_etc) {
            out += ",...";
        }

        const bool is_root = (level == 0);
        if (indent && ndim > 1) {
            if (scalar_dropped) {
                out += ",";
            }
            out += "\n";
        }

        if (scalar_dropped && is_root) {
            if (ndim > 1) {
                if (indent) {
                    out += std::string(indent_size, ' ') + "...\n";
                } else {
                    out += ",...";
                }
            } else if (!print_etc) {
                out += ",...";
            }
        }

        if (indent && ndim > 1) {
            out += indentation;
        }
        out += "]";
        return out;
    };

    return render(x, 0);
}

}  // namespace

std::string tsstr(const torch::Tensor& s) {
    if (!s.defined()) {
        throw std::invalid_argument("tsstr: tensor is undefined");
    }
    if (s.dim() != 0) {
        throw std::invalid_argument("tsstr: tensor is not scalar");
    }

    std::ostringstream ss;
    apply_format(ss);

    switch (s.scalar_type()) {
    case torch::kFloat32:
        ss << s.item<float>();
        break;
    case torch::kFloat64:
        ss << s.item<double>();
        break;
    case torch::kInt8:
        ss << static_cast<int>(s.item<int8_t>());  // avoid char printing
        break;
    case torch::kInt16:
        ss << s.item<int16_t>();
        break;
    case torch::kInt32:
        ss << s.item<int32_t>();
        break;
    case torch::kInt64:
        ss << s.item<int64_t>();
        break;
    case torch::kUInt8:
        ss << static_cast<unsigned int>(s.item<uint8_t>());
        break;
    case torch::kBool:
        ss << (s.item<bool>() ? 1 : 0);
        break;
    default:
        throw std::invalid_argument(std::string("tsstr: unsupported dtype ") +
                                    c10::toString(s.scalar_type()));
    }

    return ss.str();
}

std::string tstr(const torch::Tensor& t, bool indent) {
    constexpr int indent_size = 4;
    std::ostringstream oss;

    if (!t.defined()) {
        return "<undefined tensor>";
    }

    if (t.numel() == 0) {
        oss << "<empty>, (shape=" << t.sizes() << ", dtype=" << t.dtype() << ", dev=" << t.device()
            << ", req_grad=" << (t.requires_grad() ? "true" : "false") << ")";
        return oss.str();
    }

    torch::Tensor x = t.detach();
    if (x.is_cuda()) {
        x = x.cpu();
    }
    const int64_t dim = x.dim();

    const int64_t max_scalar_to_show = indent ? 64 : 32;
    int64_t max_scalars_to_show_per_1D_vector = max_scalar_to_show;
    if (dim > 1) {
        const int64_t n = x.sizes().back();
        const int64_t last_dim_vectors = x.numel() / std::max<int64_t>(n, 1);
        max_scalars_to_show_per_1D_vector =
            std::max<int64_t>(max_scalar_to_show / std::max<int64_t>(last_dim_vectors, 1), 1);
    }

    oss << "(shape=" << t.sizes() << ", dtype=" << t.dtype() << ", dev=" << t.device()
        << ", req_grad=" << (t.requires_grad() ? "true" : "false") << ")";
    if (indent) {
        oss << "\n" << std::string(indent_size, ' ');
    } else {
        oss << " ";
    }

    oss << render_tensor_values_compact(x, max_scalar_to_show, max_scalars_to_show_per_1D_vector, indent);
    return oss.str();
}

}  // namespace leakflow::debug

extern "C" {

[[gnu::used]] const char* pt(const torch::Tensor* t) {
    static thread_local std::string buf;
    if (!t) {
        return "Error: Tensor is null";
    }
    buf = leakflow::debug::tensor_header_str(*t);
    return buf.c_str();
}

[[gnu::used]] const char* ptv(const torch::Tensor* t) {
    static thread_local std::string buf;
    if (!t) {
        return "Error: Tensor is null";
    }
    buf = leakflow::debug::tstr(*t, /*indent=*/true);
    return buf.c_str();
}

[[gnu::used]] const char* dtv(const torch::Tensor* t) {
    static thread_local std::string buf;
    if (!t) {
        return "Error: Tensor is null";
    }
    buf = leakflow::debug::tstr(*t, /*indent=*/false);
    return buf.c_str();
}

[[gnu::used]] const char* ps(const torch::Tensor* t) {
    static thread_local std::string buf;
    if (!t) {
        return "Error: Tensor is null";
    }
    buf = leakflow::debug::sizes_str(t->sizes());
    return buf.c_str();
}

}  // extern "C"
