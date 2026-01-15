#!/bin/bash

# Search Agent

python -m agent_gem search_synthesize \
    --num_domains 100 \
    --num_entities_each_domain 10 \
    --num_tasks_each_entity 1 \
    --embedding_path /home/yofuria/PLM/Qwen3-Embedding-0.6B \
    --output ./search_output/search_output_v3.json \
    --faiss_index_path ./search_output/faiss_index_v3.index \
    --text_mapping_path ./search_output/text_mapping_v3.json \
    --max_workers 5 \
    --search_depth 10 \
    --search_breadth 30 \
    --num_iterations 1