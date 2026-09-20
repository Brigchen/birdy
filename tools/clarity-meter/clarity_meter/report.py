# -*- coding: utf-8 -*-
"""生成清晰度打分 HTML 报告（写在原图片目录，表格内嵌原图）。"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

_BROWSER_IMG_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def default_report_path(root: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(root) / f"clarity_report_{stamp}.html"


def _rel_url(rel_name: str) -> str:
    parts = Path(str(rel_name).replace("\\", "/")).parts
    return "/".join(quote(p, safe="") for p in parts if p not in (".", ""))


def _file_cell(row: Dict[str, Any]) -> str:
    name = str(row.get("name") or "")
    safe_name = html.escape(name)
    ext = Path(name).suffix.lower()
    href = _rel_url(name)
    preview = str(row.get("preview_uri") or "")
    if preview.startswith("data:image/"):
        img_src = preview
    elif href and ext in _BROWSER_IMG_EXT:
        img_src = href
    else:
        img_src = ""
    if img_src:
        if href and ext in _BROWSER_IMG_EXT:
            img = (
                f'<a href="{href}" target="_blank" rel="noopener">'
                f'<img src="{img_src}" alt="{safe_name}" />'
                f"</a>"
            )
        else:
            img = f'<img src="{img_src}" alt="{safe_name}" />'
    else:
        img = '<span class="na">无预览</span>'
    return (
        f'<td class="file">{img}'
        f'<div class="fn"><code>{safe_name}</code></div></td>'
    )


def _fmt_score(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "—"


def write_html_report(result: Dict[str, Any], out_path: Optional[str] = None) -> str:
    rows: List[Dict[str, Any]] = list(result.get("rows") or [])
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            0 if r.get("mask_score") is not None else 1,
            float(r["mask_score"]) if r.get("mask_score") is not None else 0.0,
            str(r.get("name") or ""),
        ),
    )
    min_c = float(result.get("min_clarity") or 35)
    root = str(result.get("root") or "")
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body_rows = []
    for i, r in enumerate(rows_sorted, start=1):
        verdict = str(r.get("verdict") or "")
        vclass = {
            "通过": "ok",
            "模糊": "blur",
            "未检出鸟体": "nobird",
            "读图失败": "fail",
        }.get(verdict, "")
        ms = r.get("mask_score")
        bar_w = 0
        if ms is not None:
            bar_w = max(0, min(100, int(round(float(ms)))))
        body_rows.append(
            "<tr>"
            f"<td class='n'>{i}</td>"
            f"{_file_cell(r)}"
            f"<td>{int(r.get('width') or 0)}×{int(r.get('height') or 0)}</td>"
            f"<td class='n'>{int(r.get('n_birds') or 0)}</td>"
            f"<td class='n'>{_fmt_score(r.get('bbox_score'))}</td>"
            f"<td class='score'><div class='bar'><span style='width:{bar_w}%'></span>"
            f"</div>{_fmt_score(ms)}</td>"
            f"<td class='v {vclass}'>{html.escape(verdict)}</td>"
            "</tr>"
        )
    table = "\n".join(body_rows) if body_rows else (
        "<tr><td colspan='7' class='empty'>目录中没有可评分的图片。</td></tr>"
    )
    mode = "切割图整图+掩膜" if result.get("use_full_frame") else "大图中央鸟框+掩膜"
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>清晰度打分报告</title>
<style>
  body {{ font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif; margin: 24px;
         background: #f5f5f5; color: #222; }}
  h1 {{ font-size: 1.35rem; margin: 0 0 6px; }}
  .sub {{ color: #666; margin: 0 0 18px; font-size: 0.92rem; }}
  .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 12px 16px; min-width: 88px;
           box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
  .card b {{ display: block; font-size: 1.25rem; }}
  .card span {{ color: #888; font-size: 0.8rem; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff;
           border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left;
            vertical-align: middle; }}
  th {{ background: #eef6ef; font-weight: 600; }}
  td.n, th.n {{ text-align: right; }}
  td.file img {{ display: block; max-width: 280px; max-height: 180px; border-radius: 6px;
                 object-fit: contain; background: #f0f0f0; }}
  td.file .fn {{ margin-top: 6px; }}
  td.file .fn code {{ font-size: 0.85rem; word-break: break-all; }}
  .na {{ color: #888; font-size: 0.85rem; }}
  .v.ok {{ color: #1b7a3a; font-weight: 600; }}
  .v.blur {{ color: #b42318; font-weight: 600; }}
  .v.nobird {{ color: #b54708; font-weight: 600; }}
  .v.fail {{ color: #666; }}
  .bar {{ height: 6px; background: #eee; border-radius: 3px; margin-bottom: 4px; width: 88px; }}
  .bar span {{ display: block; height: 100%; background: #2e7d32; border-radius: 3px; }}
  .empty {{ text-align: center; color: #888; padding: 24px; }}
  .hint {{ margin-top: 16px; color: #666; font-size: 0.85rem; }}
</style>
</head>
<body>
  <h1>清晰度打分报告</h1>
  <p class="sub">目录 {html.escape(root)}<br/>时间 {html.escape(when)} · 阈值 {min_c:.0f} · {html.escape(mode)} · 只读评分，未删除任何文件</p>
  <div class="cards">
    <div class="card"><b>{int(result.get('total') or 0)}</b><span>合计</span></div>
    <div class="card"><b>{int(result.get('n_pass') or 0)}</b><span>通过</span></div>
    <div class="card"><b>{int(result.get('n_blur') or 0)}</b><span>模糊</span></div>
    <div class="card"><b>{int(result.get('n_nobird') or 0)}</b><span>未检出鸟</span></div>
    <div class="card"><b>{int(result.get('n_fail') or 0)}</b><span>读图失败</span></div>
  </div>
  <table>
    <thead>
      <tr>
        <th class="n">#</th>
        <th>图片</th>
        <th>尺寸</th>
        <th class="n">鸟数</th>
        <th class="n">整框分</th>
        <th>鸟体掩膜分</th>
        <th>判定</th>
      </tr>
    </thead>
    <tbody>
      {table}
    </tbody>
  </table>
  <p class="hint">本报告与照片放在同一目录。预览上的<strong>红线</strong>是计分用的鸟体掩膜（其它检出为橙色）。点击预览打开原图。判定「模糊」表示低于阈值；本工具不会删图。</p>
</body>
</html>
"""
    out = Path(out_path) if out_path else default_report_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return str(out)
