# -*- coding: utf-8 -*-
"""
调用本机 DxO PureRAW 对 Screened_images 中的 RAW 批量降噪（无人值守）。

原理：PureRAW 自带的 Lightroom 插件通过命令行驱动主程序，可直接用
Python subprocess 调用。Lightroom Classic 必须已经在运行，否则 PureRAW
只把 Lightroom 拉起来就退出，一张都不处理。处理结束后它仍会通知
Lightroom 导入结果：

    PureRAWv6.exe --as-lightroom-last-settings-plugin
                  --lr-version="<本机 Lightroom Classic 版本>" --batch-file="<每行一个绝对路径>"

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
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_PURERAW_EXE_CANDIDATES = [
    r"C:\Program Files\DxO\DxO PureRAW 6\PureRAWv6.exe",
    r"C:\Program Files\DxO\DxO PureRAW 5\PureRAWv5.exe",
    r"C:\Program Files\DxO\DxO PureRAW 4\PureRAWv4.exe",
    r"C:\Program Files\DxO\DxO PureRAW 3\PureRAWv3.exe",
]
_DXO_SUFFIX = "-dxo"
_OUT_EXTS = {".jpg", ".jpeg", ".dng", ".tif", ".tiff"}
_POLL_INTERVAL = 3.0
_STALL_BUDGET = 1800.0  # 单帧最长等待；每有新输出重新计时


def _extra_output_roots() -> List:
    """
    预设若把输出写到完全独立的目录（不在源目录树内），
    可设置环境变量 DXO_OUTPUT_DIR 指向该目录（多个用分号分隔）。
    """
    env = os.environ.get("DXO_OUTPUT_DIR", "").strip()
    if not env:
        return []
    return [p for p in env.split(";") if p.strip()]


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


def _hidden_subprocess_kwargs() -> dict:
    """控制台程序不要弹出窗口。本机默认终端是 Windows Terminal，每次 tasklist / powershell 都会闪一个窗口。"""
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return {
        "startupinfo": startupinfo,
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
    }


def _win_pids_named(image_name: str) -> List[int]:
    """按 exe 名取 PID。用 Toolhelp，不启动 tasklist，因此不会弹出终端。"""
    if os.name != "nt" or not image_name:
        return []
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel = ctypes.windll.kernel32
    snap = kernel.CreateToolhelp32Snapshot(0x00000002, 0)
    if not snap or snap == ctypes.c_void_p(-1).value:
        return []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        want = image_name.lower()
        pids: List[int] = []
        ok = kernel.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            if entry.szExeFile.lower() == want:
                pids.append(int(entry.th32ProcessID))
            ok = kernel.Process32NextW(snap, ctypes.byref(entry))
        return pids
    finally:
        kernel.CloseHandle(snap)


def _find_running_pureraw() -> Optional[Tuple[int, str]]:
    """检测已在运行的 PureRAW 主程序实例，返回 (PID, 启动时间文本)。

    PureRAW 是单实例应用。托盘里已有的实例（常见为 --run-in-tray）
    不会接 --as-lightroom-last-settings-plugin 批处理：新进程交完参数
    就退出，队列不会开始。调用前要先关掉这个实例。
    """
    if os.name != "nt":
        return None
    exe = find_pureraw_exe()
    if exe is None:
        return None
    pids = _win_pids_named(Path(exe).name)
    if not pids:
        return None
    return pids[0], ""


def _lightroom_classic_exe() -> Optional[str]:
    """本机 Adobe Lightroom Classic 的 lightroom.exe。"""
    roots = [Path(r"C:\Program Files\Adobe"), Path(r"C:\Program Files (x86)\Adobe")]
    found: List[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for exe in root.glob("*/lightroom.exe"):
            found.append(str(exe))
    if not found:
        return None
    found.sort(key=lambda p: _pe_file_version(p), reverse=True)
    return found[0]


def _pe_file_version(path: str) -> str:
    """读 PE 的 FileVersion，如 15.5.1。读不到则返回空串。"""
    if os.name != "nt" or not path or not os.path.isfile(path):
        return ""
    import ctypes
    from ctypes import wintypes

    size = ctypes.windll.version.GetFileVersionInfoSizeW(path, None)
    if not size:
        return ""
    buf = ctypes.create_string_buffer(size)
    if not ctypes.windll.version.GetFileVersionInfoW(path, 0, size, buf):
        return ""
    lptr = ctypes.c_void_p()
    llen = wintypes.UINT()
    if not ctypes.windll.version.VerQueryValueW(
        buf, "\\VarFileInfo\\Translation", ctypes.byref(lptr), ctypes.byref(llen)
    ):
        return ""
    pair = ctypes.cast(lptr, ctypes.POINTER(ctypes.c_uint16 * 2)).contents
    key = f"\\StringFileInfo\\{pair[0]:04x}{pair[1]:04x}\\FileVersion"
    if not ctypes.windll.version.VerQueryValueW(
        buf, key, ctypes.byref(lptr), ctypes.byref(llen)
    ):
        return ""
    text = ctypes.wstring_at(lptr, llen.value) if llen.value else ""
    return text.split("\x00", 1)[0].strip()


def lightroom_version_for_pureraw() -> str:
    """PureRAW 的 --lr-version。须与已安装的 Lightroom Classic 一致，否则直接退出。"""
    exe = _lightroom_classic_exe()
    if not exe:
        raise RuntimeError(
            "未找到 Adobe Lightroom Classic。PureRAW 的批处理接口要核对它的版本，"
            "请安装 Lightroom Classic 后重试。"
        )
    ver = _lr_version_token(_pe_file_version(exe))
    if not ver:
        raise RuntimeError(f"无法读取 Lightroom Classic 版本：{exe}")
    return ver


def _lr_version_token(text: str) -> str:
    """PureRAW 只认主版本号，如 15.5.1。文件版本末尾的 .0 要去掉。"""
    match = re.search(r"(\d+\.\d+\.\d+)", text or "")
    return match.group(1) if match else (text or "").strip()


def _lightroom_is_running() -> bool:
    return bool(_win_pids_named("Lightroom.exe"))


def _lightroom_window_title() -> str:
    """Lightroom 主窗口标题。进程在、窗口还没出来时返回空串。"""
    if os.name != "nt":
        return ""
    import ctypes
    from ctypes import wintypes

    pids = set(_win_pids_named("Lightroom.exe"))
    if not pids:
        return ""
    user32 = ctypes.windll.user32
    titles: List[str] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _collect(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if int(pid.value) not in pids:
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        if buf.value:
            titles.append(buf.value)
        return True

    user32.EnumWindows(_collect, 0)
    return "\n".join(titles)


def _ensure_lightroom_running(log_cb: Callable[[str], None]) -> None:
    """插件批处理只在 Lightroom 主窗口已经出现时才会降噪。

    只看到 Lightroom.exe 进程不够：目录还在打开时，PureRAW 会以退出码 0
    结束且不调用 processImage。已经有主窗口则直接继续。
    """
    exe = _lightroom_classic_exe()
    if not exe:
        raise RuntimeError(
            "未找到 Adobe Lightroom Classic。PureRAW 的批处理要它已在运行才会降噪。"
        )
    title = _lightroom_window_title()
    if title:
        return
    if not _lightroom_is_running():
        log_cb(
            "DxO PureRAW：Lightroom Classic 未在运行。"
            "这个接口要先有 Lightroom 才会降噪，正在启动…"
        )
        subprocess.Popen([exe])
    else:
        log_cb("DxO PureRAW：Lightroom 进程已在，等待主窗口…")
    deadline = time.time() + 180
    while time.time() < deadline:
        title = _lightroom_window_title()
        if title:
            log_cb(f"DxO PureRAW：Lightroom 窗口已出现（{title.splitlines()[0][:80]}），等待目录加载…")
            time.sleep(20)
            return
        time.sleep(1)
    raise RuntimeError("已启动 Lightroom Classic，但 180 秒内没有看到主窗口。")


def _pureraw_log_hint() -> str:
    """从刚结束的 PureRAW 日志里抽出退出原因，供 Birdy 界面显示。"""
    log = Path(os.environ.get("LOCALAPPDATA", "")) / "DxO_Labs" / "DxO PureRAW 6" / "Logs" / "PureRAWv6.txt"
    if not log.is_file():
        return ""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    hints = []
    for ln in lines:
        msg = ln.split(" - ", 1)[-1]
        if any(
        key in msg
        for key in (
                "is absent",
                "Available LR version",
                "ExitCause=",
                "launchImportConfigDialog",
                "openFilesInApplication",
                "Return with result",
                "preset name is not set",
                "export file was not created",
                "not installed",
                "not supported",
                "Processing failed to launch",
                "Task bar doesn't exist",
                "processImage",
                "saveAsJPG",
                "Processing Cancelled",
                "tmpCollectionFile",
            )
        ):
            hints.append(msg.strip()[:180])
    if not hints:
        return ""
    return " PureRAW 日志：" + " | ".join(hints[-6:])


def _cli_batch_unsupported_hint() -> str:
    """PureRAW 处理开始后自行退出、任务全部被取消时，给出可执行的建议。

    实测 6.8.0.7：「Start processing」之后主进程约 30 秒就结束，保存任务队列
    被析构，已排队的图全部 "Processing Cancelled" → "Cannot save image"，
    一张都出不来。同一台机器上 PureRAW 自己的 GUI 也是闪屏后消失，DxO 的
    崩溃上报（CrashReports/__sentry-event）记录 level=fatal / native。
    所以不是命令行参数的问题，而是 PureRAW 自身在本机启动后崩溃退出。
    """
    log = Path(os.environ.get("LOCALAPPDATA", "")) / "DxO_Labs" / "DxO PureRAW 6" / "Logs" / "PureRAWv6.txt"
    if not log.is_file():
        return ""
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if "Processing Cancelled" not in text and "Cannot save image" not in text:
        return ""
    return (
        "DxO PureRAW：处理一开始主进程就自行退出，排队中的图被全部取消"
        "（日志里 Start processing → Processing Cancelled → Cannot save image）。"
        "本机实测 PureRAW 6.8 连界面都是闪屏后崩溃退出，不是参数或输出目录的问题。"
        "建议先修复安装 DxO PureRAW（或在驱动/系统更新后重装），"
        "也可以回退到 6.7 再重跑本批量。"
    )


def _available_lr_version_from_log() -> str:
    log = Path(os.environ.get("LOCALAPPDATA", "")) / "DxO_Labs" / "DxO PureRAW 6" / "Logs" / "PureRAWv6.txt"
    if not log.is_file():
        return ""
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    found = re.findall(r"Available LR version is ([0-9.]+)", text)
    return _lr_version_token(found[-1]) if found else ""


def _ensure_lightroom_plugin(log_cb: Callable[[str], None], pureraw_exe: str) -> None:
    """确保 Lightroom 的 Modules 里有 DxO 插件，且 exe 路径已写入。

    插件入口缺失或 PluginData.lua 里仍是 DXO_EXECUTABLE_PATH_PLACEHOLDER
    时，Lightroom 会报 "PureRaw is absent at ..." 并以 ExitCause=2 结束，不降噪。
    手动复制 LightroomPluginData 而没跑 DxO 安装器时就会留下占位符。
    Lightroom 已经开着时，新写入的插件要重开 Lightroom 才生效。
    """
    src_root = Path(pureraw_exe).parent / "LightroomPluginData"
    dest_root = Path(os.environ.get("APPDATA", "")) / "Adobe" / "Lightroom" / "Modules"
    if not src_root.is_dir() or not dest_root.is_dir():
        return
    names = ("DxOPureRAWv6.lrplugin", "DxOPureRAWv6.import.lrplugin")
    token = b"DXO_EXECUTABLE_PATH_PLACEHOLDER"
    # Lua 字符串里用正斜杠：反斜杠会触发 \P、\D 之类的无效转义
    exe_bytes = str(Path(pureraw_exe)).replace("\\", "/").encode("utf-8")
    copied = False
    patched = 0
    for name in names:
        src = src_root / name
        if not src.is_dir():
            continue
        dest = dest_root / name
        if not dest.exists():
            import shutil

            shutil.copytree(src, dest)
            copied = True
        for file in dest.rglob("*"):
            if not file.is_file() or file.stat().st_size > 2_000_000:
                continue
            try:
                blob = file.read_bytes()
            except OSError:
                continue
            if token in blob:
                try:
                    file.write_bytes(blob.replace(token, exe_bytes))
                    patched += 1
                except OSError as exc:
                    log_cb(f"DxO PureRAW：写入 Lightroom 插件失败（{file.name}）：{exc}")
    if not (copied or patched):
        return
    what = "已从安装目录补上" if copied else "里仍写着占位符路径"
    log_cb(
        f"DxO PureRAW：Lightroom 插件{what}，已写入 {pureraw_exe}。"
        "请关掉并重新打开 Lightroom 后再跑，否则这次仍会 ExitCause=2。"
    )


def _close_running_pureraw(log_cb: Callable[[str], None]) -> None:
    """关掉托盘里的 PureRAW，让接下来的插件批处理成为主实例。"""
    running = _find_running_pureraw()
    if running is None:
        return
    pid = running[0]
    log_cb(
        f"DxO PureRAW 已在托盘运行（PID {pid}）。"
        "这个实例不接收 Birdy 的批处理参数，先关闭再重新启动。"
    )
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=20,
            **_hidden_subprocess_kwargs(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"无法关闭已在运行的 PureRAW（PID {pid}）：{exc}"
        ) from exc
    deadline = time.time() + 20
    while time.time() < deadline:
        if _find_running_pureraw() is None:
            return
        time.sleep(0.4)
    raise RuntimeError(
        f"PureRAW（PID {pid}）关闭超时。请从托盘退出 DxO PureRAW 后重试。"
    )


def _iter_files_depth(root: Path, max_depth: int = 4):
    """遍历目录树（含子目录），限制向下深度，避免在超大相片库上全递归。"""
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        for fn in filenames:
            yield Path(dirpath) / fn


def _scan_dxo_output_index(roots) -> Dict[str, List[Path]]:
    """
    扫描若干目录树，建立 -DxO 输出文件索引 {小写文件名: [路径]}。
    PureRAW 预设的「自定义目标目录/子文件夹」会把输出写到源文件的
    子目录（甚至预设目录），因此必须沿源目录树向下搜索。
    """
    idx: Dict[str, List[Path]] = defaultdict(list)
    seen: set = set()
    for root in roots:
        if not root:
            continue
        rp = Path(root)
        if not rp.is_dir() or str(rp).lower() in seen:
            continue
        seen.add(str(rp).lower())
        for f in _iter_files_depth(rp):
            if f.suffix.lower() in _OUT_EXTS:
                n = f.name.lower()
                if _DXO_SUFFIX in n:
                    idx[n].append(f)
    return idx


def _outputs_for_source(source: str, idx: Dict[str, List[Path]]) -> List[Path]:
    """按源文件名在输出索引中匹配 <stem>-DxO*.xxx（大小写不敏感）。"""
    stem_l = Path(source).stem.lower() + _DXO_SUFFIX
    out: List[Path] = []
    for name, lst in idx.items():
        if name.startswith(stem_l):
            out.extend(lst)
    return out


def dxo_output_candidates(source: str, extra_roots=()) -> List[Path]:
    """
    预测/查找某个源文件经 PureRAW 处理后的输出路径（各种格式/命名）。
    预设命名 <stem>-DxO.xxx（或带档位后缀）；可能在源文件同目录，
    也可能在预设指定的子目录——因此沿源目录树向下搜索。
    """
    src = Path(source)
    roots = [src.parent]
    roots.extend(Path(r) for r in extra_roots)
    idx = _scan_dxo_output_index(roots)
    found = _outputs_for_source(source, idx)
    # 同名候选排序：jpg > dng > tif，其余按路径名
    rank = {".jpg": 0, ".jpeg": 0, ".dng": 1, ".tif": 2, ".tiff": 2}
    found.sort(key=lambda p: (rank.get(p.suffix.lower(), 9), p.name.lower()))
    return found


def _existing_outputs(sources: List[str]) -> set:
    """收集源列表已有的 -DxO 输出（沿各源父目录树搜索，含子目录）。"""
    roots = {Path(s).parent for s in sources if s}
    idx = _scan_dxo_output_index(list(roots) + _extra_output_roots())
    got = set()
    for s in sources:
        for cand in _outputs_for_source(s, idx):
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
    idx = _scan_dxo_output_index(
        list({Path(s).parent for s in srcs}) + _extra_output_roots()
    )
    todo = [s for s in srcs if not _outputs_for_source(s, idx)]
    pre = {str(c) for s in srcs for c in _outputs_for_source(s, idx)}
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

    # 托盘里已有的实例不接插件批处理。先关掉，再让本次启动成为主实例。
    try:
        _close_running_pureraw(_log)
    except Exception:
        batch_file.unlink(missing_ok=True)
        raise

    try:
        lr_ver = lightroom_version_for_pureraw()
        _ensure_lightroom_running(_log)
        _ensure_lightroom_plugin(_log, exe)
    except Exception:
        batch_file.unlink(missing_ok=True)
        raise
    _log(
        f"DxO PureRAW：启动批量降噪，共 {len(todo)} 张"
        f"（{Path(exe).name}，Lightroom {lr_ver}，按上次预设处理）…"
    )
    t0 = time.time()

    # 不隐藏窗口。无窗口时另一台机器会在读完预设后以 ExitCause=2 退出，processImage 不会发生。
    creationflags = 0
    launch = [
        exe,
        "--as-lightroom-last-settings-plugin",
        f"--lr-version={lr_ver}",
        f"--batch-file={batch_file}",
    ]
    proc = subprocess.Popen(launch, creationflags=creationflags)

    total = len(todo)
    last_n = len(_existing_outputs(todo))
    last_progress_t = time.time()
    proc_gone = False
    retries_left = 2
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
            if proc.poll() is not None and not proc_gone:
                proc_gone = True
                rc = proc.returncode
                if n == 0:
                    grace_until = time.time() + 20
                    while time.time() < grace_until and n == 0:
                        time.sleep(_POLL_INTERVAL)
                        n = len(_existing_outputs(todo))
                    if n > last_n:
                        last_n = n
                        last_progress_t = time.time()
                        proc_gone = True
                        _log(f"DxO PureRAW：启动进程已退出，输出仍在增加（{n}/{total}）…")
                        continue
                if n == 0 and retries_left > 0:
                    retries_left -= 1
                    proc_gone = False
                    hint = _pureraw_log_hint()
                    alt = _available_lr_version_from_log()
                    if alt and alt != lr_ver:
                        lr_ver = alt
                        launch[2] = f"--lr-version={lr_ver}"
                        _log(f"DxO PureRAW：改用日志里的 Lightroom 版本 {lr_ver} 再试。")
                    _log(
                        f"DxO PureRAW：进程已退出（退出码 {rc}），没有生成输出。"
                        "Lightroom 可能还在打开，稍后再试。"
                        + hint
                    )
                    time.sleep(10)
                    _ensure_lightroom_running(_log)
                    # PureRAW 读完批处理清单就会删掉它，重试前必须重写，
                    # 否则第二次启动只会报 "Files list for processing is empty"。
                    batch_file.write_text("\n".join(todo), encoding="utf-8")
                    proc = subprocess.Popen(launch, creationflags=creationflags)
                    last_progress_t = time.time()
                    continue
                if n == 0:
                    _log(
                        f"DxO PureRAW：进程已退出（退出码 {rc}），没有生成输出。"
                        f"本次 --lr-version={lr_ver}。"
                        + _pureraw_log_hint()
                    )
                    advice = _cli_batch_unsupported_hint()
                    if advice:
                        _log(advice)
                    break
                _log(
                    f"DxO PureRAW：启动进程已退出（退出码 {rc}，"
                    f"{len(cur)}/{total} 张有输出），继续等待剩余文件…"
                )
            if time.time() - last_progress_t > _STALL_BUDGET:
                if proc_gone:
                    _log(
                        f"DxO PureRAW：启动进程退出后长时间无新输出"
                        f"（{_STALL_BUDGET:.0f}s），终止监控。"
                        "若桌面有 PureRAW 窗口停在等待状态，请关闭后重跑。"
                    )
                else:
                    _log(
                        f"DxO PureRAW：等待超过 {_STALL_BUDGET:.0f}s 无新输出，终止。"
                    )
                if proc.poll() is None:
                    proc.terminate()
                break
    finally:
        try:
            batch_file.unlink(missing_ok=True)
        except OSError:
            pass

    outputs = sorted(_existing_outputs(srcs))
    outputs = _ensure_jpeg_outputs(outputs, srcs, _log)
    covered = sum(
        1
        for s in srcs
        if any(
            Path(o).name.lower().startswith(Path(s).stem.lower() + _DXO_SUFFIX)
            for o in outputs
        )
    )
    reported = len(srcs)
    fail = reported - covered
    # 输出格式体检：全 DNG 说明「上次设置」是 DNG 预设，提示如何切回 JPEG
    dng_n = sum(1 for o in outputs if Path(o).suffix.lower() == ".dng")
    jpg_n = sum(
        1 for o in outputs if Path(o).suffix.lower() in (".jpg", ".jpeg")
    )
    if dng_n and not jpg_n:
        _log(
            "ℹ 「上次设置」仍为 DNG 输出。DNG 已由 Birdy 转为 100% JPEG"
            "（降噪/光学校正保留）；若想让 PureRAW 直接出 JPEG，"
            "在其界面用 JPEG 预设处理一张或设为默认。"
        )
    _log(
        f"DxO PureRAW：结束，成功 {covered}/{reported}，用时 "
        f"{time.time() - t0:.0f}s。"
    )
    return {"ok": covered, "fail": fail, "total": reported, "outputs": outputs}


def _ensure_jpeg_outputs(
    outputs: List[str],
    sources: List[str],
    log_cb: Callable[[str], None],
) -> List[str]:
    """
    输出格式兜底：PureRAW 按 DNG 预设输出时（体积极大），把 DNG 渲染为
    100% JPEG（同目录同名），成功后删除 DNG。DxO 的降噪/光学校正已在
    DNG 像素里完成，这里只是换容器；渲染用 Birdy 的 RAW 显影管线。
    JPEG 已存在的源跳过。无论预设是 DNG 还是 JPEG，Birdy 拿到的都是 JPEG。
    """
    dng_out = [o for o in outputs if Path(o).suffix.lower() == ".dng"]
    if not dng_out:
        return outputs
    jpg_stems = {
        Path(o).stem for o in outputs if Path(o).suffix.lower() in (".jpg", ".jpeg")
    }
    todo_dng = [d for d in dng_out if Path(d).stem not in jpg_stems]
    if not todo_dng:
        return outputs
    src_by_stem: Dict[str, str] = {}
    for s in sources:
        src_by_stem[Path(s).stem] = s
    log_cb(
        f"⚠ PureRAW 输出了 {len(todo_dng)} 张 DNG（预设为 DNG 输出）。"
        "Birdy 将其渲染为 100% JPEG 后删除 DNG，识别不受影响…"
    )
    kept: List[str] = []
    converted = 0
    try:
        from image_io import copy_exif_from_raw_to_jpeg, read_raw_bgr_full
    except Exception:
        read_raw_bgr_full = None  # type: ignore
        copy_exif_from_raw_to_jpeg = None  # type: ignore
    import cv2

    for i, dng in enumerate(todo_dng, 1):
        stem = Path(dng).stem
        jpg_path = str(Path(dng).with_suffix(".jpg"))
        try:
            if read_raw_bgr_full is None:
                raise RuntimeError("rawpy 不可用")
            bgr = read_raw_bgr_full(dng, half_size=False)
            if bgr is None or bgr.size == 0:
                raise RuntimeError("解码为空")
            if not cv2.imwrite(
                jpg_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
            ):
                raise OSError("JPEG 写入失败")
            src = src_by_stem.get(stem.split("-DxO")[0])
            if src and copy_exif_from_raw_to_jpeg is not None:
                copy_exif_from_raw_to_jpeg(src, jpg_path)
            try:
                os.remove(dng)
            except OSError:
                pass
            converted += 1
            kept.append(jpg_path)
            log_cb(f"DxO PureRAW：DNG→JPEG {i}/{len(todo_dng)}")
        except Exception as e:
            log_cb(f"⚠ DNG→JPEG 失败 {Path(dng).name}: {e}（保留 DNG）")
            kept.append(dng)
    final = [o for o in outputs if Path(o).suffix.lower() != ".dng"]
    final.extend(kept)
    if converted:
        log_cb(
            f"DxO PureRAW：已转换 {converted}/{len(todo_dng)} 张 DNG→JPEG。"
            "要直接出 JPEG，请在 PureRAW 界面用 JPEG 预设处理一张后重试。"
        )
    return sorted(set(final))


def map_sources_to_outputs(
    sources: List[str],
) -> dict:
    """返回 {源路径: DxO 输出路径}；无输出的源不出现在结果里。

    键同时包含原样路径与 resolve 后路径，调用方可直接 get(p)。
    输出可能位于源文件所在目录树内的子目录（预设自定义目标），一次建索引。
    """
    out: dict = {}
    if not sources:
        return out
    idx = _scan_dxo_output_index(
        list({Path(s).parent for s in sources if s}) + _extra_output_roots()
    )
    rank = {".jpg": 0, ".jpeg": 0, ".dng": 1, ".tif": 2, ".tiff": 2}
    for s in sources:
        found = _outputs_for_source(s, idx)
        if found:
            found.sort(key=lambda p: (rank.get(p.suffix.lower(), 9), p.name.lower()))
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
