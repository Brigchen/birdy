# -*- coding: utf-8 -*-
"""PyQt 后台任务。"""

from __future__ import annotations

from typing import Sequence, Tuple

from PyQt5.QtCore import QThread, pyqtSignal

from .options import StabilizeOptions
from .pipeline import stabilize_file, stabilize_paths

Job = Tuple[str, str]


class StabilizeWorker(QThread):
    log_line = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        options: StabilizeOptions,
        *,
        file_job: Job | None = None,
        folder_jobs: Sequence[str] | None = None,
        output_dir: str = "",
    ):
        super().__init__()
        self._options = options
        self._file_job = file_job
        self._folder_jobs = list(folder_jobs or [])
        self._output_dir = output_dir
        self._cancel = False

    def stop(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            if self._folder_jobs:
                done, errors = stabilize_paths(
                    self._folder_jobs,
                    self._output_dir,
                    self._options,
                    log=lambda m: self.log_line.emit(m),
                    progress=lambda d, t: self.progress.emit(d, t),
                    should_cancel=lambda: self._cancel,
                )
                if self._cancel:
                    self.failed.emit("已取消")
                    return
                lines = [f"成功 {len(done)} 个"]
                lines.extend(done)
                if errors:
                    lines.append(f"失败 {len(errors)} 个")
                    lines.extend(errors)
                if done:
                    self.finished_ok.emit("\n".join(lines))
                else:
                    self.failed.emit("\n".join(lines) or "没有成功的文件")
                return

            if not self._file_job:
                self.failed.emit("没有输入")
                return
            src, dest = self._file_job
            out = stabilize_file(
                src,
                dest,
                self._options,
                log=lambda m: self.log_line.emit(m),
                progress=lambda d, t: self.progress.emit(d, t),
                should_cancel=lambda: self._cancel,
            )
            if self._cancel:
                self.failed.emit("已取消")
            else:
                self.finished_ok.emit(out)
        except InterruptedError as e:
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(str(e))
