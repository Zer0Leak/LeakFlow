#pragma once

#include "leakflow/core/caps.hpp"

#include <string>

namespace leakflow {

enum class PadDirection {
    Input,
    Output,
};

enum class PadPresence {
    Required,
    Optional,
    OnRequest,
};

// How an input pad's metadata contributes to a fused (Analyze) output.
//   Primary   -- the data the output is made from and measured on; its capture
//                facts define the output's capture identity (the default).
//   Reference -- a parameter carried from another experiment (e.g. profiling PoIs
//                or a trained model applied to attack traces); the output is
//                guided by it but is not *about* it, so its facts forward as
//                provenance (origin.<pad>.<key>) and never join the capture union.
// See docs/design/metadata_klass_taxonomy.md.
enum class PadMetadataRole {
    Primary,
    Reference,
};

class Pad {
public:
    Pad(std::string name,
        PadDirection direction,
        Caps caps,
        PadPresence presence = PadPresence::Required,
        PadMetadataRole metadata_role = PadMetadataRole::Primary);

    [[nodiscard]] const std::string& name() const;
    [[nodiscard]] PadDirection direction() const;
    [[nodiscard]] const Caps& caps() const;
    [[nodiscard]] PadPresence presence() const;
    [[nodiscard]] bool is_required() const;
    [[nodiscard]] PadMetadataRole metadata_role() const;

private:
    std::string name_;
    PadDirection direction_;
    Caps caps_;
    PadPresence presence_;
    PadMetadataRole metadata_role_;
};

} // namespace leakflow
