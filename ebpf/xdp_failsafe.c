/*
 * xdp_failsafe.c — XDP kernel program for load-balancer circuit breaker.
 *
 * Loaded via BCC from controller.py.  When lb_health_map[0] == 1 (circuit
 * open), TCP packets destined for the load balancer on port 8000 or 50051
 * are redirected to a fallback IP/MAC at the XDP layer — before the Linux
 * network stack processes them.
 *
 * Supported kernel: 5.15+ (Ubuntu 22.04 LTS).
 * NOT supported on macOS — see README.md for Lima VM instructions.
 */

#include <uapi/linux/if_ether.h>
#include <uapi/linux/ip.h>
#include <uapi/linux/tcp.h>
#include <uapi/linux/in.h>
#include <bcc/proto.h>

/* 0 = healthy (XDP_PASS), 1 = circuit open (redirect) */
BPF_ARRAY(lb_health_map,   u32, 1);

/* Big-endian fallback IPv4 address (set by controller.py) */
BPF_ARRAY(fallback_ip_map, u32, 1);

/* 6-byte fallback MAC address (set by controller.py) */
BPF_ARRAY(fallback_mac_map, u8,  6);

/* ---------- helpers -------------------------------------------------- */

static __always_inline u16 csum_fold(u32 csum)
{
    csum = (csum >> 16) + (csum & 0xffff);
    csum += (csum >> 16);
    return (u16)(~csum);
}

/* ---------- main XDP program ----------------------------------------- */

int xdp_failsafe(struct xdp_md *ctx)
{
    void *data_end = (void *)(long)ctx->data_end;
    void *data     = (void *)(long)ctx->data;

    /* ── 1. Bounds-check Ethernet header ─────────────────────────────── */
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    /* ── 2. Only handle IPv4 ─────────────────────────────────────────── */
    if (eth->h_proto != __constant_htons(ETH_P_IP))
        return XDP_PASS;

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return XDP_PASS;

    /* ── 3. Only TCP ────────────────────────────────────────────────── */
    if (ip->protocol != IPPROTO_TCP)
        return XDP_PASS;

    struct tcphdr *tcp = (void *)((u8 *)ip + (ip->ihl * 4));
    if ((void *)(tcp + 1) > data_end)
        return XDP_PASS;

    /* Only redirect traffic destined for LB HTTP (8000) or gRPC (50051) */
    u16 dport = __be16_to_cpu(tcp->dest);
    if (dport != 8000 && dport != 50051)
        return XDP_PASS;

    /* ── 4. Check circuit state ─────────────────────────────────────── */
    int zero = 0;
    u32 *health = lb_health_map.lookup(&zero);
    if (!health || *health == 0)
        return XDP_PASS;   /* circuit closed — normal forwarding */

    /* ── 5. Rewrite destination MAC ─────────────────────────────────── */
    u8 *mac = fallback_mac_map.lookup(&zero);
    if (mac) {
        /* BCC unrolls small fixed loops fine */
        eth->h_dest[0] = mac[0];
        eth->h_dest[1] = mac[1];
        eth->h_dest[2] = mac[2];
        eth->h_dest[3] = mac[3];
        eth->h_dest[4] = mac[4];
        eth->h_dest[5] = mac[5];
    }

    /* ── 6–7. Rewrite destination IP and fix checksum ───────────────── */
    u32 *fallback_ip = fallback_ip_map.lookup(&zero);
    if (!fallback_ip)
        return XDP_PASS;

    u32 old_ip = ip->daddr;
    u32 new_ip = *fallback_ip;

    /* RFC 1141 incremental checksum update */
    u32 csum = ~csum_unfold(ip->check);
    csum    += (~old_ip & 0xffff);
    csum    += (old_ip >> 16);
    csum    += (new_ip & 0xffff);
    csum    += (new_ip >> 16);
    ip->check = csum_fold(csum);

    ip->daddr = new_ip;

    /* ── 8. Transmit the rewritten packet back out the same interface ── */
    return XDP_TX;
}
