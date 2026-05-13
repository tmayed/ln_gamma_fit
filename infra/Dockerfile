FROM python_base

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies + R base
RUN apt-get update && apt-get install -y \
    # core tools
    curl \
    make \
    openssh-server \
    ca-certificates \
    gnupg \
    dirmngr \
    locales \
    && rm -rf /var/lib/apt/lists/*

# 2. Configure locale (important for R)
RUN sed -i '/en_GB.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen

ENV LANG=en_GB.UTF-8
ENV LC_ALL=en_GB.UTF-8

# 3. Prepare SSH (same as before)
RUN mkdir -p /run/sshd

# 5. Create a dedicated user for ID 1000 (so SSH has a home directory to route to)
# Find the existing user with UID 1000, delete them, and create our appuser
RUN OLD_USER=$(getent passwd 1000 | cut -d: -f1) \
    && if [ -n "$OLD_USER" ]; then userdel -f "$OLD_USER"; fi \
    && useradd -m -u 1000 -s /bin/bash appuser

RUN mkdir -p /entry /workspace

WORKDIR /entry

# Copy and setup entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Fix permissions
RUN chown -R appuser:appuser /entry
RUN chown -R appuser:appuser /workspace

# IMPORTANT: Do NOT switch to USER 1000 here. 
# We need to stay root so the entrypoint can start the SSH daemon.
# When Gemini SSHs in, it will authenticate as 'appuser' (UID 1000) automatically.

WORKDIR /workspace

ENTRYPOINT ["/entry/entrypoint.sh"]
