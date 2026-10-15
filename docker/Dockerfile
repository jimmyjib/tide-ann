# tide-ann: tide-server, tide-coordinator, tide-build-pq.
#
#   docker build -f docker/Dockerfile -t tide-ann:latest .
#
# The build stage compiles every target and runs the C++ tests (ctest); the image ships the binaries only.
# The base image is pinned by digest (ubuntu:22.04, multi-platform index) and the cppzmq header by checksum.

FROM ubuntu:22.04@sha256:962f6cadeae0ea6284001009daa4cc9a8c37e75d1f5191cf0eb83fe565b63dd7 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake libzmq3-dev ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# cppzmq (header-only C++ binding of libzmq), fetched at a fixed version and checked.
ARG CPPZMQ_VERSION=4.10.0
ARG CPPZMQ_SHA256=1f8b641161dcf12641ae4951c2c49552de425be88d66c988ec5e85046f1320f6
RUN curl -fsSL -o /usr/local/include/zmq.hpp \
        "https://raw.githubusercontent.com/zeromq/cppzmq/v${CPPZMQ_VERSION}/zmq.hpp" \
    && echo "${CPPZMQ_SHA256}  /usr/local/include/zmq.hpp" | sha256sum -c -

WORKDIR /src
COPY CMakeLists.txt ./
COPY src ./src
COPY tests ./tests
RUN cmake -S . -B build \
    && cmake --build build -j"$(nproc)" \
    && ctest --test-dir build --output-on-failure \
    && cmake --install build --prefix /opt/tide

FROM ubuntu:22.04@sha256:962f6cadeae0ea6284001009daa4cc9a8c37e75d1f5191cf0eb83fe565b63dd7

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends libzmq5 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/tide/bin/ /usr/local/bin/
WORKDIR /work
EXPOSE 28800 29699
