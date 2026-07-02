#include "leakflow/plugins/core/buffer_summary.hpp"

#include "leakflow/render/summary_renderer.hpp"

namespace leakflow::plugins::core {

std::string summarize_buffer(const Buffer& buffer, std::int64_t summary_level)
{
    return render::render_summary_plain(buffer.describe(summary_level));
}

} // namespace leakflow::plugins::core
