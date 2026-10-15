#include "coordinator_types.h"

#include <iostream>
#include <sstream>

namespace tide {
DaemonState g_daemon;
std::ostream* g_current_out = nullptr;
}  // namespace tide

int main() {
    using namespace tide;

    g_daemon.context = std::make_unique<zmq::context_t>(1);
    zmq::socket_t peer(*g_daemon.context, zmq::socket_type::pair);
    peer.bind("inproc://daemon-socket-ownership");
    g_daemon.sockets.emplace_back(*g_daemon.context, zmq::socket_type::pair);
    g_daemon.sockets[0].connect("inproc://daemon-socket-ownership");

    // Model repeated external daemon commands after the first workload.  The
    // persistent server-thread queues are the ownership marker.  A pending
    // frame must remain available to the owner thread despite any number of
    // command-boundary reset calls.
    g_daemon.main_response_queue =
        std::make_unique<ann::ThreadSafeQueue<ServerResponse>>();
    g_daemon.server_queues.emplace_back(
        std::make_unique<ann::ThreadSafeQueue<ServerWorkItem>>());
    const std::string owned = "owned";
    peer.send(zmq::buffer(owned), zmq::send_flags::none);
    for (int command = 0; command < 1000; ++command) reset_socket_state();

    zmq::message_t preserved;
    const auto received = g_daemon.sockets[0].recv(preserved, zmq::recv_flags::none);
    if (!received || preserved.to_string() != "owned") {
        std::cerr << "command-boundary reset touched an owner-thread socket\n";
        return 1;
    }

    // Once the owner threads/queues have been stopped, the direct-socket
    // fallback may safely drain stale frames at a command boundary.
    g_daemon.main_response_queue.reset();
    g_daemon.server_queues.clear();
    const std::string stale = "stale";
    peer.send(zmq::buffer(stale), zmq::send_flags::none);
    reset_socket_state();
    zmq::message_t drained;
    if (g_daemon.sockets[0].recv(drained, zmq::recv_flags::dontwait)) {
        std::cerr << "direct-socket reset failed to drain a stale frame\n";
        return 1;
    }

    g_daemon.sockets.clear();
    peer.close();
    g_daemon.context.reset();
    std::cout << "daemon socket ownership test passed\n";
    return 0;
}
