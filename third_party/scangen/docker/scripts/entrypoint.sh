#!/bin/bash
HOST_UID=$(stat -c %u /mnt/work/src)
HOST_GID=$(stat -c %g /mnt/work/src)

# If UID is 0, we're likely on Docker Desktop (Mac/Windows)
# where file permissions are handled transparently
if [ "$HOST_UID" = "0" ]; then
    exec gosu docker-user "$@"
fi

# On Linux, create a matching user
groupadd -g $HOST_GID hostgroup 2>/dev/null
useradd -o -u $HOST_UID -g $HOST_GID -m hostuser 2>/dev/null
exec gosu hostuser "$@"
