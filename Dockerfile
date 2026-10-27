# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
FROM mcr.microsoft.com/dotnet/sdk:10.0-noble@sha256:de4fe90a98845d75910b9769662465c84de1a934665c431279d1382b2bddc02b AS build
WORKDIR /src
COPY nba-clutch-agent/ ./nba-clutch-agent/
RUN dotnet restore nba-clutch-agent/AiAgents.NbaClutchAgent.Web/AiAgents.NbaClutchAgent.Web.csproj --locked-mode \
 && dotnet build nba-clutch-agent/AiAgents.NbaClutchAgent.Web/AiAgents.NbaClutchAgent.Web.csproj -c Release --no-restore \
 && dotnet test nba-clutch-agent/AiAgents.NbaClutchAgent.Web/AiAgents.NbaClutchAgent.Web.csproj -c Release --no-build --no-restore \
 && dotnet publish nba-clutch-agent/AiAgents.NbaClutchAgent.Web/AiAgents.NbaClutchAgent.Web.csproj -c Release --no-build --no-restore -o /out

FROM mcr.microsoft.com/dotnet/aspnet:10.0-noble@sha256:de0ee358cdd672f7017a003bed21e9ac9ed48a27c922a59b18175b6e82fa7fee AS aspnet
FROM python:3.12.10-slim-bookworm@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db AS runtime
# .NET 10 supports Debian 12; copy the official portable runtime, supply Debian dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends libicu72=72.1-3+deb12u1 libatomic1=12.2.0-14+deb12u1 \
    libgssapi-krb5-2=1.20.1-2+deb12u5 libkrb5-3=1.20.1-2+deb12u5 \
    libk5crypto3=1.20.1-2+deb12u5 libkrb5support0=1.20.1-2+deb12u5 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 10001 hoopslab && useradd --uid 10001 --gid 10001 --no-create-home hoopslab
COPY --from=aspnet /usr/share/dotnet /usr/share/dotnet
ENV DOTNET_ROOT=/usr/share/dotnet PATH="/usr/share/dotnet:${PATH}" \
    ASPNETCORE_ENVIRONMENT=Production PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY deployment/requirements.lock /app/deployment/requirements.lock
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --require-hashes --only-binary=:all: -r deployment/requirements.lock && python -m pip check
COPY --from=build /out/ /app/web/
COPY scripts/inference/inference_core.py scripts/inference/inference_service.py /app/scripts/inference/
COPY scripts/simulation/ /app/scripts/simulation/
COPY scripts/historical/ /app/scripts/historical/
COPY data/historical/processed/ /app/data/historical/processed/
COPY data/simulation/production_v1/ /app/data/simulation/production_v1/
COPY artifacts/inference/ /app/artifacts/inference/
COPY data/analysis/decision_2025_26_regular.csv /app/data/analysis/decision_2025_26_regular.csv
COPY deployment/start.py deployment/runtime-artifacts.json /app/deployment/
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=6s --start-period=180s --retries=3 \
 CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/ready',timeout=5).read()"
ENTRYPOINT ["python", "/app/deployment/start.py"]
