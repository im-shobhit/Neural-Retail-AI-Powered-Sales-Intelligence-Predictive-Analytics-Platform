from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

# ---------------------------------------------------------
# 1. INITIALIZE THE APP
# This creates our web server. We give it a title and description
# which will automatically generate beautiful documentation later!
# ---------------------------------------------------------
app = FastAPI(
    title="NeuralRetail API",
    description="Serving Layer for AI Sales Intelligence Platform",
    version="1.0.0"
)

# ---------------------------------------------------------
# 2. ADD OBSERVABILITY (Prometheus)
# Remember PRD Requirement F-08? We need to monitor API latency.
# These two lines automatically track how fast our API is running.
# ---------------------------------------------------------
Instrumentator().instrument(app).expose(app)

# ---------------------------------------------------------
# 3. CREATE A HEALTH CHECK ENDPOINT
# An "endpoint" is like a URL you can visit. 
# @app.get("/") means when someone visits the main URL, run this function.
# It returns a simple JSON message to prove the server is alive.
# ---------------------------------------------------------
@app.get("/")
async def root_health_check():
    """
    Basic health check to ensure the API is running.
    """
    return {
        "status": "online",
        "message": "NeuralRetail API is successfully running!",
        "environment": "development"
    }