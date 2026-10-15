# docker build -f docker/Dockerfile -t tide-ann:latest .

FROM ubuntu:22.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake libzmq3-dev ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# cppzmq header
RUN curl -fsSL -o /usr/local/include/zmq.hpp https://raw.githubusercontent.com/zeromq/cppzmq/v4.10.0/zmq.hpp

WORKDIR /src
COPY CMakeLists.txt ./
COPY src ./src
COPY tests ./tests
RUN cmake -S . -B build \
    && cmake --build build -j"$(nproc)" \
    && ctest --test-dir build --output-on-failure \
    && cmake --install build --prefix /opt/tide

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends libzmq5 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/tide/bin/ /usr/local/bin/
WORKDIR /work
EXPOSE 28800 29699
