# Convenience wrappers around docker compose for the RL Trading server.
# Every target maps to a plain `docker compose ...` command, so you can always
# run those directly (see docs/deployment/docker.md) if `make` isn't available.

.PHONY: help dirs build up down restart logs ps shell admin health prune

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

dirs: ## Create the bind-mounted dirs with the container's UID (run once, before `up`)
	mkdir -p models results logs data workspaces data/auth
	# Container runs as uid 10001; without this the app cannot write to dirs
	# Docker created as root. No-op / not needed on Docker Desktop (mac, Windows).
	-chown -R 10001:10001 workspaces data/auth 2>/dev/null || \
		echo "chown atlandi (root degilsiniz veya Docker Desktop kullaniyorsunuz)"

build: ## Build the image
	docker compose build

up: dirs ## Build (if needed) and start in the background
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

admin: ## Create an admin account (or reset a password) inside the container
	docker compose exec rltrading python scripts/create_admin.py

health: ## Curl the health endpoint from the host
	curl -fsS http://localhost:8000/health && echo

prune: ## Remove dangling images/build cache (frees disk)
	docker image prune -f && docker builder prune -f
