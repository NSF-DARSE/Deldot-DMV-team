# Build the React dashboard, then run it from the FastAPI process on Cloud Run.
FROM node:20-bookworm-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/yarn.lock ./
RUN yarn install --frozen-lockfile
COPY frontend/ ./
# Empty URL → browser calls same-origin /api (required on Cloud Run).
ENV REACT_APP_BACKEND_URL=
RUN yarn build

FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
COPY backend /app/backend
COPY --from=frontend /app/frontend/build /app/frontend/build
WORKDIR /app/backend
ENV PORT=8080
ENV MONGO_URL=memory://local
ENV DB_NAME=hencheck
ENV CORS_ORIGINS=*
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}"]
