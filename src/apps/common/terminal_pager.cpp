#include "common/terminal_pager.hpp"

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <cstdio>
#include <io.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace leakflow::apps {
namespace {

[[nodiscard]] bool stdout_is_terminal()
{
#ifdef _WIN32
    return _isatty(_fileno(stdout)) == 1;
#else
    return isatty(STDOUT_FILENO) == 1;
#endif
}

[[nodiscard]] bool terminal_supports_less()
{
    const auto* term = std::getenv("TERM");
    const std::string_view terminal_type = term != nullptr ? term : "";
    return !terminal_type.empty() && terminal_type != "dumb";
}

[[nodiscard]] std::string configured_less_argument(const TerminalPagerOptions& options)
{
    const auto* configured = std::getenv(options.less_environment_variable.c_str());
    std::string less_options = configured != nullptr ? configured : options.default_less_options;
    if (less_options.empty() || less_options.front() == '-') {
        return less_options;
    }
    return "-" + less_options;
}

[[nodiscard]] bool write_all(int fd, std::string_view output)
{
#ifdef _WIN32
    (void)fd;
    (void)output;
    return false;
#else
    auto* data = output.data();
    auto remaining = output.size();
    while (remaining > 0) {
        const auto written = ::write(fd, data, remaining);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EPIPE) {
                return true;
            }
            return false;
        }
        data += written;
        remaining -= static_cast<std::size_t>(written);
    }
    return true;
#endif
}

[[nodiscard]] bool page_with_less(std::string_view output, const TerminalPagerOptions& options)
{
#ifdef _WIN32
    (void)output;
    (void)options;
    return false;
#else
    int pipe_fds[2];
    if (::pipe(pipe_fds) != 0) {
        return false;
    }

    const auto child = ::fork();
    if (child < 0) {
        ::close(pipe_fds[0]);
        ::close(pipe_fds[1]);
        return false;
    }

    if (child == 0) {
        ::dup2(pipe_fds[0], STDIN_FILENO);
        ::close(pipe_fds[0]);
        ::close(pipe_fds[1]);

        const auto less_argument = configured_less_argument(options);
        if (less_argument.empty()) {
            ::execlp("less", "less", static_cast<char*>(nullptr));
        } else {
            ::execlp("less", "less", less_argument.c_str(), static_cast<char*>(nullptr));
        }
        _exit(127);
    }

    ::close(pipe_fds[0]);

    const auto previous_sigpipe_handler = std::signal(SIGPIPE, SIG_IGN);
    const auto wrote_output = write_all(pipe_fds[1], output);
    ::close(pipe_fds[1]);
    std::signal(SIGPIPE, previous_sigpipe_handler);

    int status = 0;
    while (::waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) {
            return false;
        }
    }

    if (!wrote_output) {
        return false;
    }

    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

} // namespace

void present_with_optional_pager(std::string_view output, const TerminalPagerOptions& options)
{
    if (options.enabled && stdout_is_terminal() && terminal_supports_less() && page_with_less(output, options)) {
        return;
    }

    std::cout << output;
}

} // namespace leakflow::apps
