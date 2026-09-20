# -*- coding: utf-8 -*-
"""后台打分（只读）。"""

from __future__ import annotations

from typing import Any, Dict

from PyQt5.QtCore import QThread, pyqtSignal

from .score import score_folder


class ScoreWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        folder: str,
        min_clarity: float = 35.0,
        use_full_frame: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.folder = folder
        self.min_clarity = float(min_clarity)
        self.use_full_frame = bool(use_full_frame)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            def on_prog(d: Dict[str, Any]) -> None:
                if d.get("kind") == "tick":
                    self.progress.emit(
                        int(d.get("done", 0)),
                        int(d.get("total", 1)),
                        str(d.get("name") or d.get("path") or ""),
                    )

            result = score_folder(
                self.folder,
                min_clarity=self.min_clarity,
                use_full_frame=self.use_full_frame,
                progress=on_prog,
                should_cancel=lambda: self._cancel,
            )
            if self._cancel:
                self.failed.emit("已取消")
                return
            if not result.get("rows"):
                self.failed.emit("目录中没有可评分的图片。")
                return
            self.finished_ok.emit(result)
        except Exception as e:
            self.failed.emit(str(e))
