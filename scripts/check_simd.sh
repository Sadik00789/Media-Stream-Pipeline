#!/usr/bin/env bash
# Inspect CPU architecture for SIMD vector extensions

echo "============================================================"
echo "  Hardware CPU Vector SIMD Capability Inspector"
echo "============================================================"

ARCH=$(uname -m)
echo "Architecture: $ARCH"

if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then
    if [ -f /proc/cpuinfo ]; then
        FLAGS=$(grep -m1 "flags" /proc/cpuinfo | cut -d: -f2)
        echo -n "AVX-512 Support: "
        if echo "$FLAGS" | grep -qw "avx512f"; then echo "✅ Supported"; else echo "❌ Not Supported"; fi

        echo -n "AVX2 Support:    "
        if echo "$FLAGS" | grep -qw "avx2"; then echo "✅ Supported"; else echo "❌ Not Supported"; fi

        echo -n "FMA Support:     "
        if echo "$FLAGS" | grep -qw "fma"; then echo "✅ Supported"; else echo "❌ Not Supported"; fi

        echo -n "SSE4.2 Support:  "
        if echo "$FLAGS" | grep -qw "sse4_2"; then echo "✅ Supported"; else echo "❌ Not Supported"; fi
    fi
elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "ARM NEON Vector Engine: ✅ Supported (Standard on ARMv8+)"
fi
echo "============================================================"
