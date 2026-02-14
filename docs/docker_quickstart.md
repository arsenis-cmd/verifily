# Verifily Docker Quickstart

Run Verifily API in a container for local or VPC deployment.

## Build

```bash
docker compose build
```

## Run (no auth)

```bash
docker compose up -d
```

Server starts at `http://localhost:8080`. Health check:

```bash
curl http://localhost:8080/health
```

## Run with API key

```bash
VERIFILY_API_KEY=my-secret-key docker compose up -d
```

All `/v1/*` endpoints now require the key:

```bash
# This will return 401:
curl http://localhost:8080/v1/pipeline -X POST -H "Content-Type: application/json" \
  -d '{"config_path": "/workspace/verifily.yaml", "plan": true}'

# This will work:
curl http://localhost:8080/v1/pipeline -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-key" \
  -d '{"config_path": "/workspace/verifily.yaml", "plan": true}'
```

Public endpoints work without auth:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/ready
curl http://localhost:8080/metrics
```

## Using data files

The compose file mounts `./workspace` to `/workspace` inside the container.
Place your configs and data files there:

```bash
mkdir -p workspace
cp examples/customer_drill/raw/* workspace/
cp your_pipeline.yaml workspace/

# Reference /workspace/ paths in API calls
curl http://localhost:8080/v1/report -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-key" \
  -d '{"dataset_path": "/workspace/eval_clean.jsonl", "schema": "sft"}'
```

## Running the demo against Docker

```bash
# Start server
docker compose up -d

# Run SDK demo (assumes server at localhost:8080)
python3 scripts/demo_sdk_customer_drill.py --base-url http://localhost:8080
```

## Stop

```bash
docker compose down
```

## Logs

```bash
docker compose logs -f verifily
```

## Interactive docs

While the server is running: `http://localhost:8080/docs`
