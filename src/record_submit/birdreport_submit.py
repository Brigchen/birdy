# -*- coding: utf-8 -*-
"""可选：将 batch JSON POST 到你自建的合规中继（官方中心无稳定公开一键接口）。"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional


def submit_batch_json_via_relay(
    batch_json_path: str,
    *,
    url: Optional[str] = None,
    bearer_token: Optional[str] = None,
    timeout_s: float = 60.0,
) -> str:
    """
    若设置环境变量 ``BIRDREPORT_SUBMIT_URL`` 与 ``BIRDREPORT_BEARER_TOKEN``，
    或将 ``url`` / ``bearer_token`` 传入，则把 JSON 原文 POST 到该 URL（需你方服务端
    完成签名、鉴权与对接观鸟记录中心）。

    返回响应体文本；未配置 URL 时抛出 ``RuntimeError``。
    """
    u = url or os.environ.get("BIRDREPORT_SUBMIT_URL", "").strip()
    tok = bearer_token or os.environ.get("BIRDREPORT_BEARER_TOKEN", "").strip()
    if not u:
        raise RuntimeError(
            "未配置中继地址：请设置环境变量 BIRDREPORT_SUBMIT_URL，"
            "或自行在观鸟记录中心 / 官方渠道补录导出的 birdreport_batch.json。"
        )
    with open(batch_json_path, "rb") as f:
        body = f.read()
    req = urllib.request.Request(
        u,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            **({"Authorization": f"Bearer {tok}"} if tok else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            return raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err_body}") from e


def load_batch_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
