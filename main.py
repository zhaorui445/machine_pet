"""电子宠物项目的开发环境检查入口；不发送网络请求。"""

import sqlite3
import sys
from importlib.metadata import version

import httpx


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("电子宠物 · 开发环境检查")
    print(f"Python：{sys.version.split()[0]}")
    print(f"解释器：{sys.executable}")
    in_venv = sys.prefix != sys.base_prefix
    print(f"独立虚拟环境：{'已启用' if in_venv else '未启用，请选择项目 .venv'}")
    print(f"HTTP 客户端：httpx {httpx.__version__}")
    print(f"配置读取：python-dotenv {version('python-dotenv')}")

    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE environment_check (value TEXT)")
        connection.execute("INSERT INTO environment_check VALUES (?)", ("ok",))
        result = connection.execute("SELECT value FROM environment_check").fetchone()
        if result != ("ok",):
            raise RuntimeError("SQLite 读写检查失败")
    finally:
        connection.close()
    print(f"记忆数据库：SQLite {sqlite3.sqlite_version}，读写正常")

    print("项目路线：本地从零训练模型，不需要 API 密钥。")
    print("基础环境检查完成。运行 check_gpu.py 可验证 GPU 训练环境。")


if __name__ == "__main__":
    main()
