# eBPF XDP Failsafe — Setup Guide

## Overview

`xdp_failsafe.c` is an XDP kernel program loaded via [BCC](https://github.com/iovisor/bcc).
`controller.py` is the userspace controller that:

1. Polls `GET /health` on the load balancer every 1 s
2. Queries Prometheus P95 latency every 5 s
3. Opens the "circuit" (sets `lb_health_map[0] = 1`) after **3 consecutive failures** or P95 > 500 ms for 3 checks
4. When the circuit is open, the XDP program rewrites destination IP + MAC of all TCP packets to port 8000/50051, redirecting them to a fallback server at the kernel level — before any userspace code runs

---

## Why This Cannot Run on macOS Natively

macOS does not have a Linux kernel.  BCC and XDP require kernel 5.15+ Linux.
Running `controller.py` on a Mac will fail at `from bcc import BPF`.

---

## Option 1 — Lima VM (Recommended for local dev on Apple Silicon / Intel Mac)

[Lima](https://lima-vm.io/) starts a lightweight Linux VM with seamless
`~home` sharing.

```bash
# Install Lima
brew install lima

# Start an Ubuntu 22.04 VM (kernel 5.15, BPF ready)
limactl start --name=ubuntu template://ubuntu-lts

# Shell into the VM
limactl shell ubuntu

# Inside the VM — install BCC
sudo apt update
sudo apt install -y bpfcc-tools python3-bpfcc python3-pip iproute2 linux-headers-$(uname -r)
sudo pip3 install requests prometheus-client

# Run the controller (replace IPs / interface as needed)
sudo python3 /path/to/ebpf/controller.py \
    --interface lima0 \
    --lb-host 192.168.5.2 \
    --fallback-ip 192.168.5.3 \
    --fallback-mac 52:55:0a:00:02:02
```

---

## Option 2 — Ubuntu 22.04 Cloud VM

Any Ubuntu 22.04+ instance on AWS, GCP, Azure, etc. works out of the box.

```bash
# Install dependencies
sudo apt update
sudo apt install -y bpfcc-tools python3-bpfcc python3-pip iproute2 \
                    linux-headers-$(uname -r)
sudo pip3 install requests prometheus-client

# Run (must be root or have CAP_NET_ADMIN + CAP_BPF)
sudo python3 ebpf/controller.py \
    --interface eth0 \
    --lb-host <LB_IP> \
    --fallback-ip <FALLBACK_IP> \
    --fallback-mac <FALLBACK_MAC>
```

---

## Option 3 — Docker (Privileged Container)

Build the image from the project root:

```bash
docker build -t ebpf-controller ebpf/

docker run --rm --privileged --network host \
    -v /sys/fs/bpf:/sys/fs/bpf \
    -v /lib/modules:/lib/modules:ro \
    -v /usr/src:/usr/src:ro \
    ebpf-controller python3 controller.py \
        --interface eth0 \
        --lb-host 172.17.0.2 \
        --fallback-ip 172.17.0.3 \
        --fallback-mac 02:42:ac:11:00:03
```

> **Note**: The host machine must still be Linux with kernel ≥ 5.15.

---

## Option 4 — Kubernetes DaemonSet

Apply `k8s-daemonset.yaml` after setting the correct `FALLBACK_IP` and
`FALLBACK_MAC` environment variables:

```bash
kubectl apply -f ebpf/k8s-daemonset.yaml
kubectl -n predictive-lb get pods -l app.kubernetes.io/component=ebpf-failsafe
```

The DaemonSet runs with `hostPID: true`, `hostNetwork: true`, and
`privileged: true` — standard requirements for XDP.

---

## Manual Test Procedure

```bash
# 1. Start the controller
sudo python3 ebpf/controller.py --interface eth0 --lb-host 172.17.0.2 \
    --fallback-ip 172.17.0.3 --fallback-mac 02:42:ac:11:00:03 &

# 2. Confirm circuit is closed
curl http://localhost:9100/metrics | grep ebpf_circuit_state

# 3. Kill the load balancer to trigger 3 consecutive health failures
docker-compose stop load_balancer

# 4. Wait ~5 seconds, then confirm circuit opened
curl http://localhost:9100/metrics | grep ebpf_circuit_state
# Expected: ebpf_circuit_state{state="open"} 1.0

# 5. Verify traffic reaches the fallback
curl http://172.17.0.3:8000/health

# 6. Restart LB; circuit auto-closes after next successful health check
docker-compose start load_balancer
sleep 5
curl http://localhost:9100/metrics | grep ebpf_circuit_state
# Expected: ebpf_circuit_state{state="closed"} 1.0

# 7. Stop the controller
kill %1
```

---

## Prometheus Metrics Exposed (port 9100)

| Metric | Labels | Description |
|--------|--------|-------------|
| `ebpf_circuit_state` | `state="open"` | 1 when circuit is open (redirecting) |
| `ebpf_circuit_state` | `state="closed"` | 1 when circuit is closed (healthy) |
