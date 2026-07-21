# Convenience wrappers around docker compose for the RL Trading server.
# Every target maps to a plain `docker compose ...` command, so you can always
# run those directly (see docs/deployment/docker.md) if `make` isn't available.

.PHONY: help build up down restart logs ps shell health prune

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

build: ## Build the image
	docker compose build

up: ## Build (if needed) and start in the background
	docker compose up -d --build

down: ## Stop and remove the container
	docker compose down

restart: ## Restart the container
	docker compose restart

logs: ## Follow container logs
	docker compose logs -f rltrading

ps: ## Show container status
	docker compose ps

shell: ## Open a shell inside the running container
	docker compose exec rltrading /bin/bash

health: ## Curl the health endpoint from the host
	curl -fsS http://localhost:8000/health && echo

prune: ## Remove dangling images/build cache (frees disk)
	docker image prune -f && docker builder prune -f
