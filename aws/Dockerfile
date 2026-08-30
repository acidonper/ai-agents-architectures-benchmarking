# Use a Python image with uv pre-installed
FROM ghcr.io/sooperset/mcp-atlassian:latest

EXPOSE 8000

# 4. Pasa los argumentos necesarios al binario (corrigiendo el puerto a 8000)
CMD ["--transport", "streamable-http", "--port", "8000", "-vv"]
