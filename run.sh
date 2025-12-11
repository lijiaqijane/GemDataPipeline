cd /home/victor/project/general_agent_bundle
source /home/victor/anaconda3/bin/activate tb

# Common env for local vLLM
export LLM_PROVIDER=vllm
export VLLM_MODEL=qwen3-4b
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=local

# Batch categories with moderate rounds to improve success rate
# for cat in "travel itinerary planning" "personal finance planning" "coding interview prep" "meal planning" "event scheduling"; do
for cat in "travel itinerary planning"; do
  python -m general_agent \
    --category "$cat" \
    --sandbox "./sandbox/$(echo "$cat" | tr ' ' '-' )" \
    --rounds 3
done
