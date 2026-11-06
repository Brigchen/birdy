# -*- coding: utf-8 -*-
"""在当前用户桌面创建 GUI 快捷方式。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .paths import tool_dir


def pythonw_executable() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        return str(exe)
    sibling = exe.with_name("pythonw.exe")
    if sibling.is_file():
        return str(sibling)
    return str(exe)


def create_desktop_shortcut() -> str:
    """创建「观鸟录像稳定.lnk」，返回快捷方式路径。"""
    if sys.platform != "win32":
        raise OSError("桌面快捷方式目前仅支持 Windows")
    py = pythonw_executable().replace("'", "''")
    wd = str(tool_dir()).replace("'", "''")
    ps = f"""
$desktop = [Environment]::GetFolderPath('Desktop')
$lnk = Join-Path $desktop '观鸟录像稳定.lnk'
$w = New-Object -ComObject WScript.Shell
$s = $w.CreateShortcut($lnk)
$s.TargetPath = '{py}'
$s.Arguments = '-m video_stab --gui'
$s.WorkingDirectory = '{wd}'
$s.Description = '观鸟录像稳定：三脚架固定 / 追踪稳定器'
$s.Save()
Write-Output $lnk
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    out = (result.stdout or "").strip().splitlines()
    path = out[-1].strip() if out else ""
    if result.returncode != 0 or not path or not Path(path).is_file():
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(err or "未能创建桌面快捷方式")
    return path
