# -*- coding: utf-8 -*-
"""子进程生成观鸟地图（避免 matplotlib 与 Qt 同进程死锁）。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict

from PyQt5.QtCore import QThread, pyqtSignal

from .paths import runtime_dir, setup_import_paths


class TrackMapWorker(QThread):
    log_line = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, kwargs: Dict[str, Any], parent=None):
        super().__init__(parent)
        self._kwargs = dict(kwargs)

    @staticmethod
    def _popen_kwargs() -> Dict[str, Any]:
        kw: Dict[str, Any] = {}
        if sys.platform == "win32":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return kw

    def run(self) -> None:
        self.log_line.emit("观鸟地图：已启动生成子进程…")
        rt = runtime_dir()
        if not (rt / "gpx_track").is_dir():
            self.failed.emit(
                "缺少运行库 birdy_runtime/。请在工具目录运行: python sync_runtime.py"
            )
            return
        try:
            with tempfile.TemporaryDirectory(prefix="birdy_trackmap_") as td:
                kin = Path(td) / "kwargs.json"
                kout = Path(td) / "result.json"
                kin.write_text(
                    json.dumps(self._kwargs, ensure_ascii=False),
                    encoding="utf-8",
                )
                cmd = [
                    sys.executable,
                    "-m",
                    "gpx_track.generate_worker",
                    str(kin),
                    str(kout),
                ]
                env = os.environ.copy()
                env["PYTHONPATH"] = str(rt) + os.pathsep + env.get("PYTHONPATH", "")
                env["BIRDY_TOOL_DIR"] = str(Path(__file__).resolve().parents[1])
                env.pop("BIRDY_AMAP_CONFIG", None)
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(rt),
                    env=env,
                    **self._popen_kwargs(),
                )
                last_ping = time.monotonic()
                while proc.poll() is None:
                    if time.monotonic() - last_ping >= 3.0:
                        self.log_line.emit(
                            "观鸟地图：仍在生成（匹配 GPX、下载底图、绘制 PNG）…"
                        )
                        last_ping = time.monotonic()
                    time.sleep(0.25)
                stdout, stderr = proc.communicate(timeout=30)
                if proc.returncode != 0:
                    err_body = (stderr or stdout or "").strip()
                    if kout.is_file():
                        try:
                            payload = json.loads(kout.read_text(encoding="utf-8"))
                            if payload.get("error"):
                                err_body = str(payload["error"]).strip()
                        except Exception:
                            pass
                    self.failed.emit(err_body or f"子进程退出码 {proc.returncode}")
                    return
                if not kout.is_file():
                    self.failed.emit("子进程未生成结果文件")
                    return
                written = json.loads(kout.read_text(encoding="utf-8"))
                if written.get("error"):
                    self.failed.emit(str(written["error"]))
                    return
                self.log_line.emit("观鸟地图：绘制完成")
                self.finished_ok.emit(written)
        except subprocess.TimeoutExpired:
            self.failed.emit("生成超时")
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")
