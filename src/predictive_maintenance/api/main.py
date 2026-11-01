from fastapi import FastAPI

from predictive_maintenance.api.routes import router

app = FastAPI(
    title="Predictive Maintenance API",
    version="0.1.0",
)

app.include_router(router)