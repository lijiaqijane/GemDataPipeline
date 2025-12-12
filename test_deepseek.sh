#!/bin/bash
cd /home/victor/project/general_agent_bundle
source /home/victor/anaconda3/bin/activate tb

# Deepseek v3.2 configuration
export LLM_PROVIDER=deepseek
export VOLCANO_MODEL=deepseek-v3-2-251201
export VOLCANO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
export VOLCANO_API_KEY=47041ffc-3c83-49ee-9d79-4f70592850d2

# Test with a simple category
python -m general_agent \
  --category "travel itinerary planning" \
  --sandbox "./sandbox/test-deepseek" \
  --rounds 1
