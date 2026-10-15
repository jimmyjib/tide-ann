// sha256_hex against the FIPS 180-4 examples.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "sha256.h"

namespace {
using namespace tide;

void check(bool ok, const char* text) {
    if (!ok) { std::fprintf(stderr, "CHECK FAILED: %s\n", text); std::abort(); }
}

void sha256_known_answers() {
    check(sha256_hex(std::string()) ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "empty message");
    check(sha256_hex(std::string("abc")) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "abc");
    check(sha256_hex(std::string("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          "two-block message");
    const std::string million(1000000, 'a');
    check(sha256_hex(million) ==
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
          "one million 'a'");
    check(sha256_hex(std::string("abd")) != sha256_hex(std::string("abc")), "one changed byte changes the digest");
}
}  // namespace

int main() {
    sha256_known_answers();
    std::printf("sha256_known_answers OK\n");
    return 0;
}
