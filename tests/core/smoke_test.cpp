#include "leakflow/core/version.hpp"

#include <iostream>
#include <string_view>

int main()
{
    constexpr std::string_view expected = "LeakFlow 0.10 build";

    if (leakflow::build_banner() != expected) {
        std::cerr << "unexpected build banner\n";
        return 1;
    }

    return 0;
}
