# -*- coding: utf-8 -*-
"""
调用本机 DxO PureRAW 对 Screened_images 中的 RAW 批量降噪（无人值守）。

原理：PureRAW 自带的 Lightroom 插件通过命令行驱动主程序；该接口不依赖
Lightroom，可直接用 Python subprocess 调用：

    PureRAWv6.exe --as-lightroom-last-settings-plugin
                  --lr-version="14.0" --batch-file="<每行一个绝对路径>"

参数说明：
  - 队列按 PureRAW 界面里「上次使用的预设」处理（降噪档位/输出格式/
    目标目录/命名均由预设决定，命令行不可传参，符合"固定参数"用法）。
  - 输出文件写在每个源文件旁（或预设的自定义目录），文件名在源名后加
    "-DxO" 后缀（如 f27601408-DxO.jpg）。

首次使用前，在 PureRAW 界面建好预设：DeepPRIME XD3 + 输出 JPEG(质量 100)
+ 自定义目标目录（推荐留空=输出到源文件旁），并对 A7R VI + 镜头下载
一次光学模组。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

_PURERAW_EXE_CANDIDATES = [
    r"C:\Program Files\DxO\DxO PureRAW 6\PureRAWv6.exe",
    r"C:\Program Files\DxO\DxO PureRAW 5\PureRAWv5.exe",
    r"C:\Program Files\DxO\DxO PureRAW 4\PureRAWv4.exe",
    r"C:\Program Files\DxO\DxO PureRAW 3\PureRAWv3.exe",
]
_DXO_SUFFIX = "-dxo"
_POLL_INTERVAL = 3.0
_STALL_BUDGET = 1800.0  # 单帧最长等待；每有新输出重新计时


def _candidate_exe_paths() -> List[str]:
    """按优先级列出可能的 PureRAW 可执行文件路径（版本无关）。"""
    cands: List[str] = []
    env = os.environ.get("DXO_PURERAW_EXE", "")
    if env:
        cands.append(env)

    roots = [r"C:\Program Files\DxO", r"C:\Program Files (x86)\DxO"]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for entry in Path(root).glob("DxO PureRAW */PureRAWv*.exe"):
            # 排除内部进程（如 PureRAWv6_saver.exe）
            if entry.stem.lower().endswith(("_saver", "_crashpad")):
                continue
            cands.append(str(entry))
    # 老兜底：固定文件名（glob 已覆盖，这里防目录命名变化）
    for p in _PURERAW_EXE_CANDIDATES:
        cands.append(p)

    if sys.platform == "darwin":
        for entry in Path("/Applications").glob("DxO PureRAW*/Contents/MacOS/*"):
            cands.append(str(entry))

    # 新版本优先（PureRAWv7 > PureRAWv6）
    cands.sort(reverse=True)
    return cands


def find_pureraw_exe() -> Optional[str]:
    """返回本机 PureRAW 可执行文件路径；未安装返回 None。

    探测顺序：环境变量 DXO_PURERAW_EXE → Program Files 下 DxO PureRAW *
    目录中的 PureRAW*.exe（任意版本）→ macOS /Applications。新装用户
    用官方默认安装即可自动发现，无需手动指定。
    """
    for p in _candidate_exe_paths():
        if p and os.path.isfile(p):
            return p
    return None


def _exe_name(exe: str) -> str:
    return Path(exe).stem.lower().replace("pureraw", "pureraw")


def dxo_output_candidates(source: str) -> List[Path]:
    """
    预测某个源文件经 PureRAW 处理后可能的输出路径（各种输出格式/命名）。
    预设命名格式为 <stem>-DxO.xxx（或带降噪档位），列在源文件同目录。
    """
    src = Path(source)
    stem = src.stem
    parent = src.parent
    out: List[Path] = []
    for ext in (".jpg", ".jpeg", ".dng", ".tif", ".tiff"):
        out.append(parent / f"{stem}-DxO{ext}")
        out.append(parent / f"{stem}{_DXO_SUFFIX.upper()}{ext}")
        out.append(parent / f"{stem}{_DXO_SUFFIX}{ext}")
    # 更精确：扫描同目录中 stem 开头且含 DxO 的文件
    if parent.is_dir():
        for f in parent.iterdir():
            n = f.name.lower()
            if n.startswith(stem.lower() + _DXO_SUFFIX) and f.suffix.lower() in (
                ".jpg", ".jpeg", ".dng", ".tif", ".tiff"
            ):
                out.append(f)
    return out


def _existing_outputs(sources: List[str]) -> set:
    got = set()
    for s in sources:
        for cand in dxo_output_candidates(s):
            if cand.is_file():
                got.add(str(cand))
    return got


def run_pureraw_batch(
    sources: List[str],
    *,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """
    批量处理一批 RAW。返回 {"ok": n_ok, "fail": n_fail, "total": n, "outputs": [...]}。

    sources：RAW 绝对路径列表（须已存在）。重复运行时，已有 -DxO 输出的
    源文件会跳过（PureRAW 预设不可覆盖时避免重复处理）。
    """
    srcs = [str(Path(s).resolve()) for s in sources if os.path.isfile(s)]
    if not srcs:
        return {"ok": 0, "fail": 0, "total": 0, "outputs": []}

    # 已有输出的源跳过（无需安装/启动 PureRAW）
    pre = _existing_outputs(srcs)
    todo = [
        s
        for s in srcs
        if not any(
            str(c) in pre for c in dxo_output_candidates(s)
        )
    ]
    if not todo:
        if log_cb is not None:
            log_cb(f"DxO PureRAW：全部 {len(srcs)} 张已有输出，跳过。")
        else:
            print(f"DxO PureRAW：全部 {len(srcs)} 张已有输出，跳过。", flush=True)
        return {
            "ok": len(srcs),
            "fail": 0,
            "total": len(srcs),
            "outputs": sorted(pre),
        }

    exe = find_pureraw_exe()
    if exe is None:
        raise RuntimeError(
            "未找到 DxO PureRAW。请安装 PureRAW 6 并在界面中配置好预设。"
        )

    def _log(msg: str) -> None:
        if log_cb is not None:
            log_cb(msg)
        else:
            print(msg, flush=True)

    batch_file = Path(os.environ.get("TEMP", ".")) / (
        f"birdy_pureraw_{int(time.time())}.txt"
    )
    batch_file.write_text("\n".join(todo), encoding="utf-8")
    _log(
        f"DxO PureRAW：启动批量降噪，共 {len(todo)} 张"
        f"（{Path(exe).name}，按上次预设处理）…"
    )
    t0 = time.time()

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        [
            exe,
            "--as-lightroom-last-settings-plugin",
            "--lr-version=14.0",
            f"--batch-file={batch_file}",
        ],
        creationflags=creationflags,
    )

    total = len(todo)
    last_n = len(_existing_outputs(todo))
    last_progress_t = time.time()
    if progress_cb:
        progress_cb(0, total, "")

    try:
        while True:
            if should_cancel is not None and should_cancel():
                _log("DxO PureRAW：收到取消请求，终止进程。")
                proc.terminate()
                break
            time.sleep(_POLL_INTERVAL)
            cur = _existing_outputs(todo)
            n = len(cur)
            if n > last_n:
                last_n = n
                last_progress_t = time.time()
                if progress_cb:
                    progress_cb(n, total, "")
                _log(f"DxO PureRAW：已完成 {n}/{total}")
            if n >= total:
                break
            if proc.poll() is not None:
                # 进程已退出；再等一小会确认没有遗漏
                time.sleep(5.0)
                cur = _existing_outputs(todo)
                if len(cur) >= total:
                    break
                _log(
                    f"DxO PureRAW：进程退出但仅 {len(cur)}/{total} 张有输出，"
                    f"视为失败 {total - len(cur)} 张。"
                )
                break
            if time.time() - last_progress_t > _STALL_BUDGET:
                _log(
                    f"DxO PureRAW：等待超过 {_STALL_BUDGET:.0f}s 无新输出，终止。"
                )
                proc.terminate()
                break
    finally:
        try:
            batch_file.unlink(missing_ok=True)
        except OSError:
            pass

    outputs = sorted(_existing_outputs(srcs))
    ok = len(outputs)
    fail = total - min(ok, total)
    _log(
        f"DxO PureRAW：结束，成功 {ok}/{total}，用时 "
        f"{time.time() - t0:.0f}s。"
    )
    return {"ok": ok, "fail": fail, "total": total, "outputs": outputs}


def map_sources_to_outputs(
    sources: List[str],
) -> dict:
    """返回 {源路径: DxO 输出路径}；无输出的源不出现在结果里。

    键同时包含原样路径与 resolve 后路径，调用方可直接 get(p)。
    """
    out: dict = {}
    for s in sources:
        found = [
            c
            for c in dxo_output_candidates(s)
            if c.is_file()
        ]
        if found:
            # 优先 jpg > dng > tif
            found.sort(
                key=lambda p: {".jpg": 0, ".jpeg": 0, ".dng": 1, ".tif": 2, ".tiff": 2}[
                    p.suffix.lower()
                ]
            )
            target = str(found[0].resolve())
            out[str(Path(s).resolve())] = target
            out[s] = target
    return out


def map_and_filter(
    image_files: List[str],
) -> Tuple[List[str], dict]:
    """
    把列表中的 RAW 替换为其 DxO 输出，并剔除输出文件自身
    （目录重扫时 -DxO.jpg 会被再次收集，须避免重复识别）。
    返回 (新列表, {源: 输出})。
    """
    m = map_sources_to_outputs(image_files)
    if not m:
        return list(image_files), m
    out_set = {str(Path(v).resolve()) for v in m.values()}
    new_list: List[str] = []
    for p in image_files:
        rp = str(Path(p).resolve())
        if rp in out_set:
            continue
        new_list.append(m.get(p, m.get(rp, p)))
    return new_list, m


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python dxo_pureraw.py <含RAW的文件夹>")
        sys.exit(1)
    folder = sys.argv[1]
    exts = {".arw", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".dng"}
    paths = [
        str(p)
        for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]
    r = run_pureraw_batch(paths)
    print(r)
