# Build the React dashboard, then run it from the FastAPI process on Cloud Run.
FROM node:20-bookworm-slim AS frontend
WORKDIR /app/oos_review/frontend
COPY oos_review/frontend/package.json oos_review/frontend/yarn.lock ./
RUN yarn install --frozen-lockfile
COPY oos_review/frontend/ ./
# Empty URL → browser calls same-origin /api (required on Cloud Run).
ENV REACT_APP_BACKEND_URL=
RUN yarn build

FROM python:3.11-slim
WORKDIR /app
COPY oos_review/backend/requirements.txt /app/oos_review/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/oos_review/backend/requirements.txt
COPY oos_review/backend /app/oos_review/backend
COPY --from=frontend /app/oos_review/frontend/build /app/oos_review/frontend/build
WORKDIR /app/oos_review/backend
ENV PORT=8080
ENV MONGO_URL=memory://local
ENV DB_NAME=hencheck
ENV CORS_ORIGINS=*
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}"]
