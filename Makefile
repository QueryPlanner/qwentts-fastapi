.PHONY: help build run stop logs test clean

help:
	@echo "Qwen3-TTS FastAPI Server"
	@echo ""
	@echo "Usage:"
	@echo "  make build    Build Docker image"
	@echo "  make run      Start the server with Docker"
	@echo "  make stop     Stop the server"
	@echo "  make logs     View server logs"
	@echo "  make test     Run test client"
	@echo "  make clean    Remove containers and volumes"

build:
	docker-compose build

run:
	docker-compose up -d
	@echo "Server starting at http://localhost:8000"
	@echo "API docs at http://localhost:8000/docs"

stop:
	docker-compose down

logs:
	docker-compose logs -f

test:
	python test_client.py

clean:
	docker-compose down -v
	rm -rf models/ test_output*.wav
