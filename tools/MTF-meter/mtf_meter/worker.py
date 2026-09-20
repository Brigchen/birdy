# -*- coding: utf-8 -*-
"""后台批量测量。"""

from __future__ import annotations

from typing import Any, Dict

from PyQt5.QtCore import QThread, pyqtSignal

from .analyze import measure_many
from .image_load import collect_images


class MeasureWorker(QThread):
    progress = pyqtSignal(int, int, str)
    row_ready = pyqtSignal(object, int, int)
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, folder: str, recursive: bool = True, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.recursive = recursive
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def was_cancelled(self) -> bool:
        return bool(self._cancel)

    def run(self) -> None:
        try:
            paths = collect_images(self.folder, recursive=self.recursive)
            if not paths:
                self.failed.emit("目录中没有 JPG/PNG/TIFF/WebP 图片。")
                return

            def on_prog(d: Dict[str, Any]) -> None:
                if d.get("kind") != "row":
                    return
                rec = d.get("row")
                done = int(d.get("done", 0))
                total = int(d.get("total", 1))
                path = str(d.get("path") or "")
                if rec is not None:
                    self.row_ready.emit(rec, done, total)
                self.progress.emit(done, total, path)

            rows = measure_many(
                paths,
                progress=on_prog,
                should_cancel=lambda: self._cancel,
            )
            self.finished_ok.emit(rows)
        except Exception as e:
            self.failed.emit(str(e))
