FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        bpfcc-tools \
        python3-bpfcc \
        python3-pip \
        linux-headers-generic \
        iproute2 \
        curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir requests prometheus-client

WORKDIR /app

COPY xdp_failsafe.c .
COPY controller.py .

# Requires --privileged, CAP_NET_ADMIN, CAP_BPF, and mounts:
#   /sys/fs/bpf:/sys/fs/bpf  /lib/modules:/lib/modules:ro  /usr/src:/usr/src:ro
#
# Example:
#   docker run --privileged --network host \
#     -v /sys/fs/bpf:/sys/fs/bpf \
#     -v /lib/modules:/lib/modules:ro \
#     -v /usr/src:/usr/src:ro \
#     ebpf-controller python3 controller.py \
#       --interface eth0 --lb-host 172.17.0.2 \
#       --fallback-ip 172.17.0.3 --fallback-mac 02:42:ac:11:00:03

CMD ["python3", "controller.py", "--help"]
