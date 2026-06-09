#!/bin/sh
# DeepSeek compatibility patch for FireCrawl
# Runs at container startup before the main app
set -e

echo "[patch] Copying helpers.js..."
cp /docker/helpers.js /app/dist/src/lib/deepseek-helpers.js

echo "[patch] Patching generic-ai.js..."
sed -i 's/if (provider === "openai" && modelName.startsWith("o3-mini"))/if (provider === "openai")/' /app/dist/src/lib/generic-ai.js

echo "[patch] Done"
