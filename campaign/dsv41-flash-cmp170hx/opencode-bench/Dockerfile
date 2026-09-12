FROM local/glm53-build-smoke:20260909
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates git ripgrep >/dev/null 2>&1 && rm -rf /var/lib/apt/lists/*
COPY opencode /usr/local/bin/opencode
RUN chmod +x /usr/local/bin/opencode && mkdir -p /root/.config/opencode /work
COPY opencode.json /root/.config/opencode/opencode.json
COPY fixture /work
COPY run-tasks.sh /run-tasks.sh
RUN chmod +x /run-tasks.sh
ENV OPENCODE_DISABLE_PROJECT_CONFIG=1 OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1
WORKDIR /work
ENTRYPOINT ["/run-tasks.sh"]
