from fastapi import FastAPI


app = FastAPI(title="Intelligent Cockpit Travel Agent")


def build_response(message: str, data: dict) -> dict:
    return {
        "code": 200,
        "message": message,
        "data": data,
    }


@app.get("/")
async def root() -> dict:
    return build_response(
        "success",
        {
            "project": "intelligent-cockpit-travel",
            "stage": "T1 backend startup",
            "status": "running",
        },
    )


@app.get("/health")
async def health() -> dict:
    return build_response(
        "success",
        {
            "service": "backend",
            "status": "ok",
        },
    )
