FROM python:3.10-bookworm

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        openjdk-17-jre-headless \
        unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md MANIFEST.in mkdocs.yml sonar-project.properties .sonarcloud.properties /workspace/
COPY .github /workspace/.github
COPY benchmarks /workspace/benchmarks
COPY data /workspace/data
COPY docs /workspace/docs
COPY src /workspace/src
COPY tests /workspace/tests
COPY tools /workspace/tools

RUN python -m pip install --upgrade pip \
    && pip install -e ".[dev,torch,stats]"

CMD ["bash"]
