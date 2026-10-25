#!/usr/bin/env bash
# vm.sh <a|b> create|status|ip|ssh|stop|start|delete   (one H100 per VM; names rpoc-a / rpoc-b)
# Stopped VMs are not billed for compute (the disk still is). Stop VMs with `vm.sh <w> stop`,
# never from inside the guest: Nebius treats a guest-initiated power-off as a failure.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
need nebius; need jq
W="${1:?usage: vm.sh <a|b> <cmd>}"; CMD="${2:?}"; shift 2
NAME="rpoc-$W"; ENVF="$HERE/.$W.env"; [ -f "$ENVF" ] && source "$ENVF"

refresh_ip() {
  local ip
  ip=$(nebius compute instance get --id "$INSTANCE_ID" --format json \
       | jq -r '.status.network_interfaces[0].public_ip_address.address // empty' | cut -d/ -f1)
  if [ -n "$ip" ]; then
    sed -i.bak '/^INSTANCE_IP=/d' "$ENVF"; rm -f "$ENVF.bak"; printf 'INSTANCE_IP="%s"\n' "$ip" >> "$ENVF"
  fi
  echo "${ip:-}"
}

case "$CMD" in
  create)
    [ -f "$ENVF" ] && die "$ENVF exists; delete the VM first"
    [ -f "$SSH_KEY_PUB" ] || die "no ssh public key at $SSH_KEY_PUB"
    SUBNET_ID=$(nebius vpc subnet list --format jsonpath='{.items[0].metadata.id}')
    [ -n "$SUBNET_ID" ] || die "no subnet in this project"
    USER_DATA=$(printf '#cloud-config\nusers:\n  - name: %s\n    sudo: ALL=(ALL) NOPASSWD:ALL\n    shell: /bin/bash\n    groups: [docker]\n    ssh_authorized_keys:\n      - %s\n' "$SSH_USER" "$(cat "$SSH_KEY_PUB")")
    echo "creating $NAME ($NEBIUS_PLATFORM $NEBIUS_PRESET, on-demand) in subnet $SUBNET_ID"
    INSTANCE_ID=$(nebius compute instance create --name "$NAME" \
      --resources-platform "$NEBIUS_PLATFORM" --resources-preset "$NEBIUS_PRESET" \
      --boot-disk-managed-disk-name "$NAME-disk" --boot-disk-managed-disk-type "$NEBIUS_BOOT_DISK_TYPE" \
      --boot-disk-managed-disk-size-gibibytes "$NEBIUS_BOOT_DISK_GIB" \
      --boot-disk-managed-disk-block-size-bytes 4096 \
      --boot-disk-managed-disk-source-image-family-image-family "$NEBIUS_IMAGE_FAMILY" \
      --boot-disk-managed-disk-source-image-family-parent-id "$NEBIUS_IMAGE_PARENT" \
      --boot-disk-attach-mode READ_WRITE --cloud-init-user-data "$USER_DATA" \
      --network-interfaces "[{\"name\":\"eth0\",\"subnet_id\":\"$SUBNET_ID\",\"ip_address\":{},\"public_ip_address\":{}}]" \
      --format jsonpath='{.metadata.id}')
    printf 'INSTANCE_ID="%s"\n' "$INSTANCE_ID" > "$ENVF"
    IP=""
    for _ in $(seq 1 60); do IP=$(refresh_ip); [ -n "$IP" ] && break; sleep 5; done
    echo "$NAME id=$INSTANCE_ID ip=${IP:-<none yet>}" ;;
  status) nebius compute instance get --id "$INSTANCE_ID" --format json | jq '{name:.metadata.name,state:.status.state,ip:.status.network_interfaces[0].public_ip_address.address}' ;;
  ip)     refresh_ip ;;
  ssh)    ssh_vm "$(refresh_ip)" "$@" ;;
  stop)   nebius compute instance stop --id "$INSTANCE_ID"; echo "stopped (disk still bills)" ;;
  start)  nebius compute instance start --id "$INSTANCE_ID"; echo "ip: $(refresh_ip)" ;;
  delete) read -r -p "delete $NAME ($INSTANCE_ID) and its disk? type yes: " a; [ "$a" = yes ] || exit 1
          nebius compute instance delete --id "$INSTANCE_ID"; rm -f "$ENVF"
          echo "check leftover disks: nebius compute disk list" ;;
  *) die "usage: vm.sh <a|b> create|status|ip|ssh|stop|start|delete" ;;
esac
