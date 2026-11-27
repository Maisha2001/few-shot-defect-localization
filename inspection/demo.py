"""Local browser workspace for comparing defect maps and recording human annotations."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import secrets

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .reviews import Store
from .serve import Engine, create_app


def workspace(engine, store, token):
    api = create_app(engine, store, token)

    @api.get("/cases/{case_id}/siblings")
    def siblings(case_id: str):
        original = store.get(case_id)
        if original is None:
            raise HTTPException(404, "Unknown case")
        with closing(store.connect()) as db:
            rows = db.execute("SELECT id,version FROM cases WHERE source_hash=? ORDER BY id",
                              (original["source_hash"],)).fetchall()
        versions = {v:arm for arm,v in engine.versions.items()}
        return {versions[r["version"]]:r["id"] for r in rows if r["version"] in versions}

    @api.get("/cases/{case_id}/annotation")
    def latest_annotation(case_id: str):
        with closing(store.connect()) as db:
            row = db.execute("""SELECT c.revision,r.payload FROM cases c LEFT JOIN reviews r
                                ON r.case_id=c.id AND r.revision=c.revision WHERE c.id=?""", (case_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Unknown case")
        fields = json.loads(row["payload"]) if row["payload"] else {"label":None,"runs":[]}
        return {"revision":row["revision"], "label":fields["label"], "runs":fields["runs"]}

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1","localhost","testserver"])
    app.mount("/api", api)
    app.mount("/", StaticFiles(directory=Path(__file__).with_name("ui"), html=True))
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8026)
    args = parser.parse_args()
    token = os.environ.get("INSPECTION_TOKEN") or secrets.token_urlsafe(32)
    app = workspace(Engine(args.model), Store(args.database), token)
    print(f"Open http://127.0.0.1:{args.port}/", flush=True)
    if "INSPECTION_TOKEN" not in os.environ:
        print(f"Local access key: {token}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)

if __name__=="__main__":
    main()
