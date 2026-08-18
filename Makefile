.PHONY: dev backend frontend

dev:
	@trap 'kill 0' INT TERM EXIT; \
	(cd backend && uv run uvicorn main:app --reload) & \
	(cd frontend && npm run dev) & \
	wait

backend:
	cd backend && uv run uvicorn main:app --reload

frontend:
	cd frontend && npm run dev

