#!/bin/bash
set -e

KEY_DIR="/workspace/.ssh"
KEY_FILE="$KEY_DIR/id_ed25519"

mkdir -p "$KEY_DIR"

# 1. Generate SSH keys if they don't exist in the shared volume
if [ ! -f "$KEY_FILE" ]; then
    echo "Generating new SSH key pair in shared volume..."
    ssh-keygen -t ed25519 -f "$KEY_FILE" -q -N ""
    
    # Give ownership of the keys to the appuser
    chown -R appuser:appuser "$KEY_DIR"
fi

# 2. Authorize the generated public key for the appuser
echo "Configuring authorized_keys for appuser..."
mkdir -p /home/appuser/.ssh
cp "${KEY_FILE}.pub" /home/appuser/.ssh/authorized_keys

# Lock down permissions (SSH will reject the login if these are wrong)
chown -R appuser:appuser /home/appuser/.ssh
chmod 700 /home/appuser/.ssh
chmod 600 /home/appuser/.ssh/authorized_keys

# 3. Start the SSH daemon in the foreground
# This replaces 'tail -f /dev/null' and keeps the container running
echo "Starting SSH daemon..."
exec /usr/sbin/sshd -D