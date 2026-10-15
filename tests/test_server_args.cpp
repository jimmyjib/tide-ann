// tide-server start-up argument checks on the real binary (argv[1]).
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

int failures = 0;

struct Run {
    int exit_code = -1;
    std::string err;
};

Run run_server(const std::string& binary, const std::vector<std::string>& args) {
    int pipe_fd[2];
    if (pipe(pipe_fd) != 0) { std::perror("pipe"); std::exit(2); }
    const pid_t child = fork();
    if (child < 0) { std::perror("fork"); std::exit(2); }
    if (child == 0) {
        dup2(pipe_fd[1], STDERR_FILENO);
        const int null_fd = ::open("/dev/null", O_WRONLY);
        if (null_fd >= 0) dup2(null_fd, STDOUT_FILENO);
        close(pipe_fd[0]);
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(binary.c_str()));
        for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
        argv.push_back(nullptr);
        execv(binary.c_str(), argv.data());
        _exit(127);
    }
    close(pipe_fd[1]);
    Run run;
    char buffer[4096];
    ssize_t got = 0;
    while ((got = read(pipe_fd[0], buffer, sizeof buffer)) > 0) run.err.append(buffer, static_cast<size_t>(got));
    close(pipe_fd[0]);
    int status = 0;
    waitpid(child, &status, 0);
    run.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    return run;
}

void expect(const std::string& binary, const char* what, const std::vector<std::string>& args, int exit_code,
            const std::string& message) {
    const Run run = run_server(binary, args);
    const bool ok = run.exit_code == exit_code && run.err.find(message) != std::string::npos;
    std::printf("%s %s: exit %d, stderr contains \"%s\"%s\n", ok ? "OK  " : "FAIL", what, run.exit_code,
                message.c_str(), ok ? "" : (" (stderr: " + run.err + ")").c_str());
    if (!ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <tide-server>\n", argv[0]); return 2; }
    const std::string server = argv[1];
    const std::string dir = "/nonexistent-test-data-dir";
    const std::string four = "tcp://s0:1,tcp://s1:1,tcp://s2:1,tcp://s3:1";
    expect(server, "missing --num-servers", {"--data-dir", dir, "--rank", "0", "--full-vectors"}, 1,
           "--data-dir, --rank and --num-servers are required");
    expect(server, "rank not below K", {"--data-dir", dir, "--rank", "4", "--num-servers", "4", "--full-vectors"}, 1,
           "is not below --num-servers 4");
    expect(server, "K - 1 peers (own address left out)",
           {"--data-dir", dir, "--rank", "0", "--num-servers", "4", "--peers", "tcp://s1:1,tcp://s2:1,tcp://s3:1",
            "--full-vectors"}, 1, "--peers names 3 servers, --num-servers 4");
    expect(server, "K + 1 peers", {"--data-dir", dir, "--rank", "0", "--num-servers", "4", "--peers",
                                   four + ",tcp://s4:1", "--full-vectors"}, 1, "--peers names 5 servers, --num-servers 4");
    expect(server, "--pq with --full-vectors", {"--data-dir", dir, "--rank", "0", "--num-servers", "4", "--pq",
                                                "--full-vectors"}, 1, "--pq and --full-vectors are exclusive");
    // Accepted arguments: the start proceeds to the data load, which fails here.
    expect(server, "exactly K peers", {"--data-dir", dir, "--rank", "1", "--num-servers", "4", "--peers", four,
                                       "--full-vectors"}, 1, "cannot load");
    expect(server, "no peers (non-P2P)", {"--data-dir", dir, "--rank", "1", "--num-servers", "4", "--pq"}, 1,
           "cannot load");
    expect(server, "unknown option", {"--data-dir", dir, "--rank", "0", "--num-servers", "4", "--no-such-option", "8",
                                      "--pq"}, 1, "Usage:");
    if (failures) { std::fprintf(stderr, "%d server argument checks failed\n", failures); return 1; }
    std::printf("server argument tests passed\n");
    return 0;
}
