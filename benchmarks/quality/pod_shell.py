"""SSH-first setup and visible stages, inside each pod's hard timeout."""


def ssh_first():
    return """set -euo pipefail
run_stage() {
  stage_name=$1
  shift
  printf 'stage_start %s %s\\n' "$stage_name" "$(/usr/bin/date -u +%FT%TZ)"
  /usr/bin/df -h /dr151-cache
  if "$@"; then stage_status=0; else stage_status=$?; fi
  printf 'stage_end %s status=%s %s\\n' "$stage_name" "$stage_status" "$(/usr/bin/date -u +%FT%TZ)"
  /usr/bin/df -h /dr151-cache
  return "$stage_status"
}
/usr/bin/cat > /dr151-ssh-setup.sh <<'DR151_SSH'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
/usr/bin/apt-get update
/usr/bin/apt-get install -y --no-install-recommends openssh-server
: "${PUBLIC_KEY:?RunPod must inject PUBLIC_KEY for SSH access}"
/usr/bin/mkdir -p /run/sshd /root/.ssh
/usr/bin/chmod 755 /run/sshd
/usr/bin/chmod 700 /root/.ssh
umask 077
printf '%s\\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
/usr/bin/chmod 600 /root/.ssh/authorized_keys
/usr/bin/ssh-keygen -l -f /root/.ssh/authorized_keys > /dev/null
/usr/bin/ssh-keygen -A
/usr/sbin/sshd -t
/usr/sbin/sshd
printf 'SSH_READY port=22 authorized_keys_mode=600\\n'
DR151_SSH
run_stage ssh /bin/bash /dr151-ssh-setup.sh
printf 'stage_start environment %s\\n' "$(/usr/bin/date -u +%FT%TZ)"
/usr/bin/df -h /dr151-cache
"""
