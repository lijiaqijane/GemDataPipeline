#!/bin/bash

# Search Agent

python -m agent_gem search_synthesize \
    --num_domains 10 \
    --num_tasks_each_entity 10 \
    --num_entities_each_domain 10 \
    --embedding_path /home/yofuria/PLM/Qwen3-Embedding-0.6B \
    --output ./search_output/search_output_v1.json \
    --faiss_index_path ./search_output/faiss_index.index \
    --text_mapping_path ./search_output/text_mapping.json