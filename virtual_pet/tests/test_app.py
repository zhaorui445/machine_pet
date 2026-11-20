import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import Pet, Server, Store, UserError, load_config


class FakeLocalRunner:
    def __init__(self):
        self.calls = []

    def reply(self, history, user, controller, memories=(), **_kwargs):
        self.calls.append((list(history), user, list(memories)))
        return controller.handle(history, user) or "本地测试回复。"


class PetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "test.sqlite3"
        self.store = Store(self.path)
        self.pet = Pet(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def test_explicit_memory_persists_and_can_be_edited(self):
        self.pet.chat("记住：我喜欢天文学")
        state = Store(self.path).state()
        self.assertEqual(state["memories"][0]["text"], "我喜欢天文学")
        mid = state["memories"][0]["id"]
        self.store.memory("我最近喜欢篮球", mid)
        self.assertIn("篮球", self.pet.chat("你记住了什么？")["reply"])
        self.assertEqual(len(self.store.state()["memories"]), 1)

    def test_demo_is_labelled_as_offline_and_does_not_use_a_model(self):
        result = self.pet.chat("你能做什么")
        self.assertEqual(result["source"], "demo")
        self.assertIn("离线演示", result["reply"])
        self.assertEqual(len(self.store.state()["messages"]), 2)

    def test_reminder_survives_restart_and_becomes_due(self):
        self.pet.chat("5秒后提醒我喝水")
        due = self.store.state()["reminders"][0]["due"]
        with patch("app.time.time", return_value=due + 1):
            state = Store(self.path).state()
        self.assertLess(state["reminders"][0]["due"], state["now"])
        self.assertFalse(state["reminders"][0]["acknowledged"])

    def test_invalid_reminders_rejected(self):
        for due in (time.time() - 60, float("nan"), float("inf")):
            with self.assertRaises(UserError):
                self.store.reminder("test", due)
        with self.assertRaises(UserError):
            self.pet.chat("0秒后提醒我喝水")
        self.assertEqual(self.store.state()["reminders"], [])

    def test_local_runner_receives_history_and_only_opted_in_memories(self):
        runner = FakeLocalRunner()
        self.pet = Pet(self.store, model_runner=runner)
        self.store.memory("我喜欢天文学")
        self.assertEqual(self.pet.chat("你好")["source"], "local_model")
        self.assertEqual(runner.calls[-1][2], [])
        with self.store.connect() as db:
            db.execute("UPDATE settings SET value='true' WHERE key='send_memory'")
        self.pet.chat("再聊聊")
        self.assertEqual(runner.calls[-1][2], ["我喜欢天文学"])
        self.assertEqual(len(self.store.state()["messages"]), 4)
        self.assertEqual(self.pet.state()["model"]["mode"], "local")

    def test_local_state_machine_memory_updates_sync_to_memory_panel(self):
        runner = FakeLocalRunner()
        self.pet = Pet(self.store, model_runner=runner)
        self.store.memory("喜欢天文学")
        result = self.pet.chat("我明确同意保存“喜欢机器人”")
        self.assertIn("已按", result["reply"])
        values = {row["text"] for row in self.store.state()["memories"]}
        self.assertIn("喜欢天文学", values)
        self.assertIn("喜欢机器人", values)

    def test_old_api_config_and_environment_are_ignored(self):
        config_path = Path(self.temp.name) / "config.json"
        config_path.write_text(json.dumps({"api_key": "must-not-load", "model": "remote"}), encoding="utf-8")
        self.assertEqual(load_config(config_path), {})
        self.assertEqual(load_config(Path(self.temp.name) / "missing.json"), {})

    def test_busy_chat_does_not_duplicate(self):
        self.pet.chat_lock.acquire()
        try:
            with self.assertRaisesRegex(UserError, "上一条消息"):
                self.pet.chat("你好")
        finally:
            self.pet.chat_lock.release()
        self.assertEqual(self.store.state()["messages"], [])


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.pet = Pet(Store(Path(self.temp.name) / "test.sqlite3"))
        self.server = Server(("127.0.0.1", 0), self.pet)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, path, body=None, headers=None):
        request = Request(self.base + path, data=json.dumps(body).encode() if body is not None else None, headers={"X-Pet-Token": self.server.token, "Content-Type": "application/json", **(headers or {})})
        with urlopen(request, timeout=3) as response:
            return response.read()

    def test_static_and_private_file_boundary(self):
        self.assertIn(self.server.token, self.request("/").decode())
        self.assertIn(b"pet-scene", self.request("/style.css"))
        for path in ("/config.json", "/data/pet.sqlite3", "/../.env"):
            with self.assertRaises(HTTPError) as result:
                self.request(path)
            self.assertEqual(result.exception.code, 404)

    def test_api_access_restrictions(self):
        for headers in ({"X-Pet-Token": "wrong"}, {"Origin": "https://unrelated.example"}, {"Host": "unrelated.example"}):
            with self.assertRaises(HTTPError) as result:
                self.request("/api/memory", {"text": "should not be stored"}, headers)
            self.assertEqual(result.exception.code, 403)
        self.assertEqual(self.pet.store.state()["memories"], [])

    def test_memory_settings_reminder_and_clear_workflow(self):
        self.request("/api/settings", {"name": "小圆", "style": "study", "send_memory": True})
        self.request("/api/memory", {"text": "喜欢读书"})
        mid = self.pet.store.state()["memories"][0]["id"]
        self.request("/api/memory", {"id": mid, "text": "喜欢科幻小说"})
        self.assertEqual(self.pet.store.state()["memories"][0]["text"], "喜欢科幻小说")
        self.request("/api/reminder", {"text": "喝水", "due": time.time() + 60})
        reminder_id = self.pet.store.state()["reminders"][0]["id"]
        self.request("/api/reminder/ack", {"id": reminder_id})
        self.assertTrue(self.pet.store.state()["reminders"][0]["acknowledged"])
        self.request("/api/chat", {"text": "你好"})
        self.request("/api/messages/clear", {})
        self.request("/api/memory/delete", {"id": mid})
        state = json.loads(self.request("/api/state"))
        self.assertEqual(state["settings"]["name"], "小圆")
        self.assertEqual(state["messages"], [])
        self.assertEqual(state["memories"], [])
        self.assertEqual(len(state["reminders"]), 1)

    def test_bad_input_is_rejected(self):
        for path, body in (("/api/chat", {"text": " "}), ("/api/chat", {"text": "x" * 2001}), ("/api/settings", {"name": "团团", "style": "unknown", "send_memory": True}), ("/api/memory", {"text": "x" * 501}), ("/api/reminder", {"text": "过期", "due": 0})):
            with self.assertRaises(HTTPError) as result:
                self.request(path, body)
            self.assertEqual(result.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
