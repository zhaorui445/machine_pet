"""Independent virtual pet. Python 3.11+, standard library only."""
from __future__ import annotations

import argparse
import json
import re
import secrets
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
STYLES = {"gentle", "brief", "study"}


class UserError(Exception):
    pass


def load_config(path: Path) -> dict:
    """Read a local, non-secret configuration object; network model settings are ignored."""
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError) as exc:
            raise UserError("config.json 无法读取，请检查 JSON 格式。") from exc
        if not isinstance(raw, dict):
            raise UserError("config.json 必须是一个 JSON 对象。")
        return {key: value for key, value in raw.items() if key in {"ui"}}
    return {}


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, text TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL, source TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY, text TEXT NOT NULL, due REAL NOT NULL, acknowledged INTEGER NOT NULL DEFAULT 0);
            """)
            for key, value in {"name": "团团", "style": "gentle", "send_memory": False}.items():
                db.execute("INSERT OR IGNORE INTO settings VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def state(self) -> dict:
        with self.connect() as db:
            return {
                "settings": {r["key"]: json.loads(r["value"]) for r in db.execute("SELECT * FROM settings")},
                "memories": [dict(r) for r in db.execute("SELECT * FROM memories ORDER BY id DESC")],
                "messages": [dict(r) for r in db.execute("SELECT * FROM (SELECT * FROM messages ORDER BY id DESC LIMIT 100) ORDER BY id")],
                "reminders": [dict(r) for r in db.execute("SELECT * FROM reminders ORDER BY due")],
                "now": time.time(),
            }

    def memory(self, text: str, memory_id: int | None = None):
        with self.connect() as db:
            if memory_id is None:
                if db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] >= 100:
                    raise UserError("最多保存 100 条记忆，请先整理已有记录。")
                db.execute("INSERT INTO memories(text, created) VALUES (?, ?)", (text, time.time()))
            elif db.execute("UPDATE memories SET text=? WHERE id=?", (text, memory_id)).rowcount == 0:
                raise UserError("这条记忆已不存在。")

    def reminder(self, text: str, due: float):
        if not time.time() < due <= time.time() + 366 * 86400:
            raise UserError("提醒时间须在未来一年以内。")
        with self.connect() as db:
            db.execute("INSERT INTO reminders(text, due) VALUES (?, ?)", (text, due))

    def exchange(self, message: str, reply: str, source: str):
        with self.connect() as db:
            db.executemany("INSERT INTO messages(role,content,source,created) VALUES (?,?,?,?)", [("user", message, source, time.time()), ("assistant", reply, source, time.time())])

    def sync_memories(self, values):
        wanted = set(values)
        with self.connect() as db:
            existing = {row["text"] for row in db.execute("SELECT text FROM memories")}
            for value in existing - wanted:
                db.execute("DELETE FROM memories WHERE text=?", (value,))
            for value in wanted - existing:
                if db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] >= 100:
                    raise UserError("记忆已达 100 条上限，请先整理后再添加。")
                db.execute("INSERT INTO memories(text, created) VALUES (?, ?)", (value, time.time()))


def clean_text(value, max_length=2000):
    if not isinstance(value, str) or not value.strip():
        raise UserError("请输入内容。")
    value = value.strip()
    if len(value) > max_length:
        raise UserError(f"内容最多 {max_length} 个字。")
    return value


class Pet:
    def __init__(self, store: Store, config: dict | None = None, model_runner=None):
        self.store = store
        self.model_runner = model_runner
        self.chat_lock = threading.Lock()

    @property
    def configured(self):
        return self.model_runner is not None

    def state(self):
        state = self.store.state()
        state["model"] = {
            "configured": self.configured,
            "mode": "local" if self.configured else "demo",
            "name": "本地 v6r5 候选" if self.configured else "离线演示",
        }
        return state

    def chat(self, message: str):
        if not self.chat_lock.acquire(blocking=False):
            raise UserError("上一条消息还在回复中，请稍等。")
        try:
            state = self.store.state()
            source = "local"
            remember = re.fullmatch(r"记住[：:]\s*(.+)", message, re.S)
            remind = re.fullmatch(r"(\d+)\s*(秒|分钟|小时)后提醒我\s*(.+)", message, re.S)
            if remember:
                content = clean_text(remember.group(1), 500)
                self.store.memory(content)
                reply = f"已保存到本机记忆：{content}。你可以在记忆面板修改或删除。"
            elif message.rstrip("？?。") in ("你记住了什么", "你记住了我什么"):
                reply = "现在保存了：\n" + "\n".join("· " + r["text"] for r in state["memories"]) if state["memories"] else "还没有保存记忆。你可以说‘记住：我喜欢天文学’。"
            elif remind:
                seconds = int(remind.group(1)) * {"秒": 1, "分钟": 60, "小时": 3600}[remind.group(2)]
                content = clean_text(remind.group(3), 200)
                self.store.reminder(content, time.time() + seconds)
                reply = f"已设置提醒：{remind.group(1)}{remind.group(2)}后，{content}。请保持服务和页面运行。"
            elif self.configured:
                from petlm.behavior_state import BehaviorController

                history = []
                local_turns = [r for r in state["messages"] if r["source"] == "local_model"]
                for index in range(0, len(local_turns) - 1, 2):
                    if local_turns[index]["role"] == "user" and local_turns[index + 1]["role"] == "assistant":
                        history.append((local_turns[index]["content"], local_turns[index + 1]["content"]))
                controller = BehaviorController()
                controller.state.long_term_memory = {row["text"]: row["text"] for row in state["memories"]}
                memories = [row["text"] for row in state["memories"][:20]] if state["settings"]["send_memory"] else ()
                reply = self.model_runner.reply(
                    history[-6:], message, controller, memories,
                    name=state["settings"]["name"], style=state["settings"]["style"],
                )
                if controller.last_source == "memory_state_machine":
                    self.store.sync_memories(controller.state.long_term_memory)
                source = "local_model"
            else:
                source = "demo"
                reply = "我在这里陪你。当前是离线演示，还没有加载本地模型；你仍可以试试‘记住：我喜欢天文学’或‘1分钟后提醒我喝水’。"
            self.store.exchange(message, reply, source)
            return {"reply": reply, "source": source}
        finally:
            self.chat_lock.release()


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, pet):
        super().__init__(address, Handler)
        self.pet = pet
        self.token = secrets.token_urlsafe(32)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        # Do not print chats, model responses or credentials to the console.
        pass

    def send_data(self, status, body, content_type="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def allowed(self):
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            return False
        origin = self.headers.get("Origin")
        return not origin or origin in {"http://" + host for host in hosts}

    def do_GET(self):
        if not self.allowed():
            return self.send_data(403, {"error": "仅允许本机页面访问。"})
        route = urlsplit(self.path).path
        if route == "/api/state":
            if self.headers.get("X-Pet-Token") != self.server.token:
                return self.send_data(403, {"error": "请刷新页面后重试。"})
            return self.send_data(200, self.server.pet.state())
        assets = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
        if route not in assets:
            return self.send_data(404, {"error": "页面不存在。"})
        name, mime = assets[route]
        body = (ROOT / "static" / name).read_text(encoding="utf-8")
        if name == "index.html":
            body = body.replace("__PET_TOKEN__", self.server.token)
        self.send_data(200, body, mime)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16000:
                raise UserError("请求大小不符合要求。")
            self.connection.settimeout(10)
            raw = self.rfile.read(length)
            # Drain a bounded request before rejecting it, avoiding a TCP reset on Windows.
            if not self.allowed() or self.headers.get("X-Pet-Token") != self.server.token:
                return self.send_data(403, {"error": "请从本机页面操作，或刷新页面后重试。"})
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise UserError("请求内容格式错误。")
            result = self.action(urlsplit(self.path).path, data)
            self.send_data(200, result or {"ok": True})
        except (UserError, ValueError, TypeError, KeyError, OverflowError) as exc:
            self.send_data(400, {"error": str(exc) if isinstance(exc, UserError) else "请求内容格式错误，请检查后重试。"})
        except Exception:
            self.send_data(500, {"error": "本地服务处理失败，请稍后重试。"})

    def action(self, route, data):
        pet = self.server.pet
        store = pet.store
        if route == "/api/chat":
            return pet.chat(clean_text(data.get("text")))
        if route == "/api/settings":
            name = clean_text(data.get("name"), 12)
            style = data.get("style")
            if style not in STYLES or type(data.get("send_memory")) is not bool:
                raise UserError("宠物设置无效。")
            with store.connect() as db:
                for key, value in {"name": name, "style": style, "send_memory": data["send_memory"]}.items():
                    db.execute("UPDATE settings SET value=? WHERE key=?", (json.dumps(value, ensure_ascii=False), key))
        elif route == "/api/memory":
            mid = int(data["id"]) if data.get("id") is not None else None
            store.memory(clean_text(data.get("text"), 500), mid)
        elif route == "/api/memory/delete":
            with store.connect() as db:
                db.execute("DELETE FROM memories WHERE id=?", (int(data["id"]),))
        elif route == "/api/reminder":
            store.reminder(clean_text(data.get("text"), 200), float(data["due"]))
        elif route in ("/api/reminder/delete", "/api/reminder/ack"):
            with store.connect() as db:
                sql = "DELETE FROM reminders WHERE id=?" if route.endswith("delete") else "UPDATE reminders SET acknowledged=1 WHERE id=?"
                db.execute(sql, (int(data["id"]),))
        elif route == "/api/messages/clear":
            with store.connect() as db:
                db.execute("DELETE FROM messages")
        else:
            raise UserError("操作不存在。")


def main():
    parser = argparse.ArgumentParser(description="独立虚拟宠物，不训练模型，不修改父项目。")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--local-model", action="store_true", help="加载已冻结的 v6r5 本地候选")
    args = parser.parse_args()
    try:
        if args.local_model:
            from local_model import LocalPetModel
            model_runner = LocalPetModel()
        else:
            model_runner = None
        pet = Pet(Store(args.data_dir / "pet.sqlite3"), model_runner=model_runner)
        server = Server(("127.0.0.1", args.port), pet)
    except (UserError, OSError) as exc:
        parser.exit(1, f"启动失败：{exc}\n")
    except (ImportError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(1, f"本地模型未能启动：{exc}\n")
    print(f"虚拟宠物已启动：http://127.0.0.1:{server.server_address[1]}", flush=True)
    print("模式：本地 v6r5 候选" if pet.configured else "模式：离线演示（使用 --local-model 加载冻结的 v6r5 候选）", flush=True)
    print("在浏览器打开上面的地址；按 Ctrl+C 停止。提醒需要服务和页面持续运行。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
