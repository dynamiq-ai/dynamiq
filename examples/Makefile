run-unstructured-local:
	docker run -p 8000:8000 -e UNSTRUCTURED_MEMORY_FREE_MINIMUM_MB=0 -d --rm --name unstructured-api downloads.unstructured.io/unstructured-io/unstructured-api:latest --port 8000 --host 0.0.0.0

stop-unstructured-local:
	docker stop unstructured-api

run-chroma-local:
	docker pull chromadb/chroma
	docker run -p 8090:8000 chromadb/chroma
