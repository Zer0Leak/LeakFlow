#include "leakflow/core/element.hpp"

#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace leakflow {
namespace {

bool has_pad_named(const std::vector<Pad>& pads, std::string_view name)
{
    for (const auto& pad : pads) {
        if (pad.name() == name) {
            return true;
        }
    }

    return false;
}

bool has_pad_template_named(const std::vector<Pad>& pads, PadDirection direction, std::string_view name)
{
    for (const auto& pad : pads) {
        if (pad.direction() == direction && pad.name() == name) {
            return true;
        }
    }

    return false;
}

bool has_property_named(const std::vector<PropertySpec>& specs, std::string_view name)
{
    for (const auto& spec : specs) {
        if (spec.name == name) {
            return true;
        }
    }

    return false;
}

const PropertySpec& property_spec_named(const std::vector<PropertySpec>& specs, std::string_view name)
{
    for (const auto& spec : specs) {
        if (spec.name == name) {
            return spec;
        }
    }

    throw std::invalid_argument("unknown property");
}

const Pad* requestable_pad_template_named(
    const std::vector<Pad>& pads,
    PadDirection direction,
    std::string_view name)
{
    for (const auto& pad : pads) {
        if (pad.direction() != direction || pad.presence() != PadPresence::OnRequest) {
            continue;
        }
        if (pad.name() == name || metadata_pad_template_matches(pad.name(), name)) {
            return &pad;
        }
    }

    return nullptr;
}

} // namespace

Element::Element(std::string name)
    : name_(std::move(name))
{
    add_property(element_name_property_spec(name_));
}

const std::string& Element::name() const
{
    return name_;
}

bool Element::name_locked() const
{
    return name_locked_;
}

const std::string& Element::element_type() const
{
    return element_type_;
}

const std::string& Element::element_kclass() const
{
    return element_kclass_;
}

int Element::provenance_slots() const
{
    return provenance_slots_;
}

bool Element::is_live() const
{
    return live_source_;
}

bool Element::is_live_driven() const
{
    return live_driven_;
}

bool Element::is_thread_boundary() const
{
    return thread_boundary_;
}

bool Element::at_end_of_stream() const
{
    return false;
}

const std::vector<Pad>& Element::pad_templates() const
{
    return pad_templates_;
}

const std::vector<Pad>& Element::input_pads() const
{
    return input_pads_;
}

const std::vector<Pad>& Element::output_pads() const
{
    return output_pads_;
}

void Element::add_pad_template(Pad pad)
{
    if (has_pad_template_named(pad_templates_, pad.direction(), pad.name())) {
        throw std::invalid_argument("duplicate pad template name");
    }

    pad_templates_.push_back(std::move(pad));
}

void Element::add_input_pad(Pad pad)
{
    if (pad.direction() != PadDirection::Input) {
        throw std::invalid_argument("input pad declaration requires PadDirection::Input");
    }
    if (has_pad_named(input_pads_, pad.name())) {
        throw std::invalid_argument("duplicate input pad name");
    }

    input_pads_.push_back(std::move(pad));
}

void Element::add_output_pad(Pad pad)
{
    if (pad.direction() != PadDirection::Output) {
        throw std::invalid_argument("output pad declaration requires PadDirection::Output");
    }
    if (has_pad_named(output_pads_, pad.name())) {
        throw std::invalid_argument("duplicate output pad name");
    }

    output_pads_.push_back(std::move(pad));
}

bool Element::can_request_input_pad(std::string_view name) const
{
    return requestable_pad_template_named(pad_templates_, PadDirection::Input, name) != nullptr;
}

bool Element::can_request_output_pad(std::string_view name) const
{
    return requestable_pad_template_named(pad_templates_, PadDirection::Output, name) != nullptr;
}

bool Element::can_request_pad(std::string_view name) const
{
    return can_request_input_pad(name) || can_request_output_pad(name);
}

bool Element::request_input_pad(std::string_view name)
{
    if (has_pad_named(input_pads_, name)) {
        return true;
    }

    const auto* pad_template = requestable_pad_template_named(pad_templates_, PadDirection::Input, name);
    if (pad_template == nullptr) {
        return false;
    }

    add_input_pad(Pad(std::string(name), PadDirection::Input, pad_template->caps()));
    return true;
}

bool Element::request_output_pad(std::string_view name)
{
    if (has_pad_named(output_pads_, name)) {
        return true;
    }

    const auto* pad_template = requestable_pad_template_named(pad_templates_, PadDirection::Output, name);
    if (pad_template == nullptr) {
        return false;
    }

    add_output_pad(Pad(std::string(name), PadDirection::Output, pad_template->caps()));
    return true;
}

void Element::add_property(PropertySpec spec)
{
    validate_property_spec(spec);
    if (has_property_named(property_specs_, spec.name)) {
        throw std::invalid_argument("duplicate property name");
    }

    // Optional properties start null; default_value is the type exemplar.
    properties_.emplace(spec.name, spec.optional ? PropertyValue{std::monostate{}} : spec.default_value);
    property_specs_.push_back(std::move(spec));
}

void Element::configure_from_descriptor(const ElementDescriptor& descriptor)
{
    set_element_identity(descriptor.type_name, descriptor.klass);
    provenance_slots_ = descriptor.provenance_slots;
    live_source_ = descriptor.live_source;
    thread_boundary_ = descriptor.thread_boundary;
    for (const auto& pad_template : descriptor.pad_templates) {
        add_pad_template(pad_template);
    }
    for (const auto& pad : descriptor.input_pads) {
        add_input_pad(pad);
    }
    for (const auto& pad : descriptor.output_pads) {
        add_output_pad(pad);
    }
    for (const auto& property : descriptor.property_specs) {
        if (property.name == "name") {
            continue;
        }
        add_property(property);
    }
}

const std::vector<PropertySpec>& Element::property_specs() const
{
    return property_specs_;
}

const std::map<std::string, PropertyValue>& Element::properties() const
{
    return properties_;
}

bool Element::has_property(std::string_view name) const
{
    return properties_.contains(std::string(name));
}

const PropertyValue& Element::property(std::string_view name) const
{
    return properties_.at(std::string(name));
}

void Element::validate_property_change(std::string_view name, const PropertyValue& value) const
{
    const auto& spec = property_spec_named(property_specs_, name);
    if (!spec.writable) {
        throw std::invalid_argument("property is read-only");
    }
    validate_property_value(spec, value);
    if (name == "name") {
        if (name_locked_) {
            throw std::invalid_argument("element name cannot be changed after adding to a pipeline");
        }
        const auto* new_name = std::get_if<std::string>(&value);
        if (new_name == nullptr || new_name->empty()) {
            throw std::invalid_argument("element name cannot be empty");
        }
    }
}

void Element::set_property(std::string_view name, PropertyValue value)
{
    validate_property_change(name, value);
    if (name == "name") {
        const auto* new_name = std::get_if<std::string>(&value);
        name_ = *new_name;
        properties_["name"] = name_;
        property_changed(name);
        return;
    }

    properties_[std::string(name)] = std::move(value);
    property_changed(name);
}

void Element::start()
{
}

std::optional<Buffer> Element::process_inputs(ElementInputs inputs)
{
    if (inputs.empty()) {
        return process(std::nullopt);
    }

    if (inputs.size() == 1) {
        return process(std::move(inputs.begin()->second));
    }

    throw std::invalid_argument("element does not support multiple input pads");
}

ElementOutputs Element::process_pads(ElementInputs inputs)
{
    auto output = process_inputs(std::move(inputs));
    ElementOutputs outputs;
    if (!output) {
        return outputs;
    }

    if (output_pads_.size() == 1) {
        outputs.emplace(output_pads_.front().name(), std::move(*output));
    } else if (output_pads_.empty()) {
        // Terminal element that still returns a buffer (e.g. a Summary with no
        // downstream link). Surface it under an empty key so the executor can
        // capture it as the run() result without routing it to any pad.
        outputs.emplace(std::string(), std::move(*output));
    } else {
        throw std::logic_error("multi-output element must override process_pads");
    }

    return outputs;
}

void Element::stop()
{
}

void Element::set_stop_token(std::stop_token token)
{
    stop_token_ = std::move(token);
}

void Element::set_live_driven(bool live_driven)
{
    live_driven_ = live_driven;
    live_driven_changed();
}

const std::stop_token& Element::stop_token() const
{
    return stop_token_;
}

bool Element::can_replay() const
{
    return true;
}

log::LogRecord Element::make_log_record(
    log::LogLevel level,
    std::string component,
    std::string message) const
{
    return {
        .level = level,
        .component = std::move(component),
        .message = std::move(message),
        .element = element_type_,
        .element_name = name_,
        .element_kclass = element_kclass_,
    };
}

void Element::set_element_identity(std::string type_name, std::string kclass)
{
    element_type_ = std::move(type_name);
    element_kclass_ = std::move(kclass);
}

void Element::set_read_only_property(std::string_view name, PropertyValue value)
{
    const auto& spec = property_spec_named(property_specs_, name);
    if (spec.writable) {
        throw std::logic_error("internal read-only property update requires a read-only property");
    }
    validate_property_value(spec, value);
    properties_[std::string(name)] = std::move(value);
}

void Element::property_changed(std::string_view)
{
}

void Element::live_driven_changed()
{
}

void Element::lock_name()
{
    name_locked_ = true;
}

} // namespace leakflow
