FROM vllm/vllm-openai:v0.11.1

# Install audio dependencies
RUN uv pip install --system vllm[audio]==0.11.1
