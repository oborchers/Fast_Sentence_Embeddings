#!/bin/bash
# EC2 user data for a paid fse measurement on AWS (Amazon Linux 2023, m8a.8xlarge).
# The launcher replaces the __PLACEHOLDERS__ before RunInstances. The pinned image of
# pins.json runs the unchanged bootstrap; Docker gives it the exact limits the
# allocation's machine check expects. The bootstrap is fetched by commit URL and
# checked by SHA-256, because user data is limited to 16 KB.
set -euo pipefail
MODE="__MODE__"  # measure | rehearse
IMAGE="python:3.13@sha256:f23cf6c01ace0da9aa4d04fd078dba0fcce0c23cf3320329a38a2b6890761f9c"

dnf install -y docker
systemctl start docker
docker pull "$IMAGE"

# Host facts for the record and for the exact cpu_model string of the machine check.
mkdir -p /fse-host
lscpu > /fse-host/lscpu.txt
grep -m1 '^model name' /proc/cpuinfo > /fse-host/cpu-model.txt
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > /fse-host/governor.txt 2>/dev/null || true

if test "$MODE" = rehearse; then
  # Matches rehearsal_machine_check: at most 2 CPUs and 6 GiB.
  limits=(--cpus=2 --cpuset-cpus=0-1 --memory=6442450944 -e MEASURE_REHEARSAL=1)
else
  # Matches machine_check: 32 CPUs, quota 32, memory exactly 128000000000 bytes.
  limits=(--cpus=32 --cpuset-cpus=0-31 --memory=128000000000)
fi

docker run -d --name fse-measure "${limits[@]}" \
  --shm-size=8g --ulimit nofile=1048576:1048576 -p 2222:22 \
  -e MEASURE_ALLOCATED_AT="__ALLOCATED_AT__" \
  -e MEASURE_ALLOCATION="/measurement-source/benchmarks/config/__ALLOCATION__" \
  -e MEASURE_STUDY="__STUDY__" \
  -e MEASURE_BILLING_PROFILE="__BILLING_PROFILE__" \
  -e PUBLIC_KEY="__PUBLIC_KEY__" \
  -e FSE_BOOTSTRAP_URL="__BOOTSTRAP_URL__" \
  -e FSE_BOOTSTRAP_SHA256="__BOOTSTRAP_SHA256__" \
  "$IMAGE" /bin/bash -c 'curl -sf "$FSE_BOOTSTRAP_URL" -o /fse-bootstrap.sh && echo "$FSE_BOOTSTRAP_SHA256  /fse-bootstrap.sh" | sha256sum -c - && exec /bin/bash /fse-bootstrap.sh'
