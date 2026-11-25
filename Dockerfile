# The proxy as a container: docker build -t compache . && docker run -p 8787:8787 -v compache:/data compache
FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml README.md LICENCE.md ./
COPY src ./src
RUN pip install --no-cache-dir .
# What it learns: sizes, times and usage, and no text.
VOLUME /data
EXPOSE 8787
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8787/compache/health', timeout=3)"
ENTRYPOINT ["compache", "proxy", "--host", "0.0.0.0"]
CMD ["--learn", "/data"]
