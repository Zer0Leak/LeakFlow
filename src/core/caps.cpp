#include "leakflow/core/caps.hpp"

#include <utility>

namespace leakflow {

Caps::Caps(std::string type)
    : type_(std::move(type))
{
}

Caps::Caps(std::string type, Params params)
    : type_(std::move(type))
    , params_(std::move(params))
{
}

const std::string& Caps::type() const
{
    return type_;
}

bool Caps::has_param(std::string_view key) const
{
    return params_.contains(std::string(key));
}

const std::string& Caps::param(std::string_view key) const
{
    return params_.at(std::string(key));
}

std::string Caps::param_or(std::string_view key, std::string default_value) const
{
    const auto found = params_.find(std::string(key));
    if (found == params_.end()) {
        return default_value;
    }

    return found->second;
}

void Caps::set_param(std::string key, std::string value)
{
    params_[std::move(key)] = std::move(value);
}

const Caps::Params& Caps::params() const
{
    return params_;
}

std::string Caps::to_string() const
{
    std::string text = type_;

    for (const auto& [key, value] : params_) {
        text += "; ";
        text += key;
        text += '=';
        text += value;
    }

    return text;
}

bool caps_are_compatible(const Caps& source, const Caps& sink)
{
    if (caps_are_any(source) || caps_are_any(sink)) {
        return true;
    }

    if (source.type() != sink.type() && sink.type() != generic_buffer_caps_type) {
        return false;
    }

    for (const auto& [key, sink_value] : sink.params()) {
        const auto source_value = source.params().find(key);
        if (source_value != source.params().end() && source_value->second != sink_value) {
            return false;
        }
    }

    return true;
}

bool caps_are_any(const Caps& caps)
{
    return caps.type() == any_caps_type;
}

} // namespace leakflow
