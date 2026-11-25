"""SQLite persistence for model outputs, the uncertainty queue and human reviews."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from datetime import datetime,timezone

class Conflict(Exception):
    pass

class Store:
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(exist_ok=True,parents=True)
        with closing(self.connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY, source_hash TEXT NOT NULL, version TEXT NOT NULL,
                    width INTEGER NOT NULL, height INTEGER NOT NULL, image BLOB NOT NULL,
                    score REAL NOT NULL, threshold REAL NOT NULL, uncertainty REAL NOT NULL,
                    heat BLOB NOT NULL, revision INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS queue ON cases(revision,uncertainty DESC,id);
                CREATE TABLE IF NOT EXISTS reviews (
                    seq INTEGER PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id),
                    revision INTEGER NOT NULL, request_id TEXT NOT NULL, payload TEXT NOT NULL,
                    created TEXT NOT NULL, UNIQUE(case_id,revision), UNIQUE(case_id,request_id));
            """)

    def connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.row_factory=sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def get(self,case_id, image=False):
        columns="image" if image else "id,source_hash,version,width,height,score,threshold,uncertainty,heat,revision"
        with closing(self.connect()) as db:
            row=db.execute(f"SELECT {columns} FROM cases WHERE id=?",(case_id,)).fetchone()
        return dict(row) if row else None

    def add(self,row):
        with closing(self.connect()) as db, db:
            db.execute("""INSERT INTO cases(id,source_hash,version,width,height,image,score,threshold,uncertainty,heat,created)
                          VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING""",(*row,datetime.now(timezone.utc).isoformat()))
        return self.get(row[0])

    def queue(self,limit):
        with closing(self.connect()) as db:
            return [dict(r) for r in db.execute("""SELECT id,version,score,uncertainty,width,height FROM cases
                       WHERE revision=0 ORDER BY uncertainty DESC,id LIMIT ?""",(limit,))]

    def review(self,case_id,request_id,expected,label,runs):
        payload=json.dumps({"expected_revision":expected,"label":label,"runs":runs},separators=(",",":"))
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            item=db.execute("SELECT revision FROM cases WHERE id=?",(case_id,)).fetchone()
            if not item:raise KeyError(case_id)
            old=db.execute("SELECT seq,revision,payload FROM reviews WHERE case_id=? AND request_id=?",(case_id,request_id)).fetchone()
            if old:
                if old["payload"]!=payload:raise Conflict("Request ID was used for a different annotation")
                return {"seq":old["seq"],"revision":old["revision"]}
            if item["revision"]!=expected:raise Conflict("Annotation revision has changed")
            rev=expected+1
            cursor=db.execute("INSERT INTO reviews(case_id,revision,request_id,payload,created) VALUES(?,?,?,?,?)",
                              (case_id,rev,request_id,payload,datetime.now(timezone.utc).isoformat()))
            db.execute("UPDATE cases SET revision=? WHERE id=?",(rev,case_id))
            return {"seq":cursor.lastrowid,"revision":rev}

    def changes(self,after,limit):
        with closing(self.connect()) as db:
            rows=db.execute("""SELECT r.seq,r.case_id,r.revision,r.created,r.payload,c.source_hash,c.version,c.width,c.height
                               FROM reviews r JOIN cases c ON c.id=r.case_id WHERE r.seq>? ORDER BY r.seq LIMIT ?""",
                            (after,limit)).fetchall()
        result=[]
        for row in rows:
            obj=dict(row);obj.update(json.loads(obj.pop("payload")));result.append(obj)
        return result
