#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
豆包动物识别 API 集成模块
支持：在线鸟类物种识别、置信度评分、错误处理

集成说明：
- 默认可配置多模型轮换（models），按日 total_tokens 统计；单模型用量 ≥ token_switch_ratio×日限额时换下一模型（默认 200 万 × 0.75）
- 非鸟主体统一归档为「其它」（不再细分人像/其它动物）
- 使用方舟 OpenAI 兼容接口 POST .../api/v3/chat/completions（不再使用易 404 的 /responses）
- 默认请求间隔 1.0s（约 1 张/秒），减轻 429
- 返回格式与本地模型兼容，并含 subject_type、archive_root_cn、archive_tag_cn（豆包）

作者: brigchen@gmail.com
版权说明: 基于开源协议，请勿商用
"""

import os
import json
import base64
import random
import threading
import requests
import time
import re
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path

# 方舟视觉模型 ID（与控制台推理接入点 Endpoint ID 一致；可改为 ep- 开头）
ARK_VISION_MODEL_DEFAULT = "doubao-seed-2-0-lite-260215"

# 多模型轮换默认列表（与方舟控制台 Endpoint 一致；1.5-pro 主型号已下线，pro-32k 仍可用）
DEFAULT_VISION_MODEL_CANDIDATES = [
    "doubao-seed-2-0-lite-260215",
    "doubao-1-5-vision-pro-32k-250115",
    "doubao-seed-2-0-mini-260215",
]

# 主体类型 → 归档一级目录（中文）
SUBJECT_TYPE_TO_ARCHIVE_ROOT = {
    "bird": "鸟类",
    "human": "人像",
    "other_animal": "其它动物",
    "other": "其它",
}


class DoubaoTokenUsageTracker:
    """按自然日统计各模型 total_tokens，超过 switch_ratio × 日限额则轮换。"""

    def __init__(
        self,
        state_path: str,
        daily_limit: int = 2_000_000,
        switch_ratio: float = 0.75,
    ):
        self.state_path = state_path
        self.daily_limit = max(1, int(daily_limit))
        self.switch_ratio = min(0.99, max(0.1, float(switch_ratio)))
        self._lock = threading.Lock()
        self._data: Dict = {}

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d")

    def _ensure_today(self) -> None:
        if self._data.get("date") != self._today():
            self._data = {"date": self._today(), "models": {}}

    def load(self) -> None:
        with self._lock:
            try:
                if os.path.isfile(self.state_path):
                    with open(self.state_path, "r", encoding="utf-8") as f:
                        self._data = json.load(f)
                else:
                    self._data = {}
            except Exception:
                self._data = {}
            self._ensure_today()

    def persist(self) -> None:
        with self._lock:
            self._ensure_today()
            try:
                d = os.path.dirname(self.state_path)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(self.state_path, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠ 豆包用量统计保存失败: {e}")

    def usage(self, model_id: str) -> int:
        with self._lock:
            self._ensure_today()
            return int(
                self._data.get("models", {}).get(model_id, {}).get("total_tokens", 0)
            )

    def add_tokens(self, model_id: str, n: int) -> None:
        if n <= 0:
            return
        with self._lock:
            self._ensure_today()
            self._data.setdefault("models", {})
            self._data["models"].setdefault(model_id, {"total_tokens": 0})
            self._data["models"][model_id]["total_tokens"] += int(n)

    def pick_model(self, candidates: List[str]) -> str:
        """在 candidates 中选取今日用量未达 switch_ratio×日限额的模型；否则选用量最少者。"""
        if not candidates:
            return ARK_VISION_MODEL_DEFAULT
        uniq = []
        for m in candidates:
            m = (m or "").strip()
            if m and m not in uniq:
                uniq.append(m)
        if not uniq:
            return ARK_VISION_MODEL_DEFAULT
        threshold = int(self.daily_limit * self.switch_ratio)
        with self._lock:
            self._ensure_today()
            for mid in uniq:
                u = int(
                    self._data.get("models", {})
                    .get(mid, {})
                    .get("total_tokens", 0)
                )
                if u < threshold:
                    print(
                        f"豆包模型选用: {mid} （今日已累计 {u}/{self.daily_limit} tokens，"
                        f"切换阈值 {threshold}）"
                    )
                    return mid
            best = min(
                uniq,
                key=lambda m: int(
                    self._data.get("models", {})
                    .get(m, {})
                    .get("total_tokens", 0)
                ),
            )
            bu = int(
                self._data.get("models", {})
                .get(best, {})
                .get("total_tokens", 0)
            )
            print(
                f"⚠ 所列模型均已≥{int(self.switch_ratio * 100)}% 日限额({threshold} tokens)，"
                f"改用今日用量最少: {best}（已 {bu}）"
            )
            return best


class DoubaoBirdAPIClient:
    """豆包鸟类识别 API 客户端"""

    @staticmethod
    def _normalize_model_candidates(
        models: Optional[List[str]], model: Optional[str]
    ) -> List[str]:
        """
        规范化候选模型列表：
        - 优先使用配置文件中的 models（保持原顺序，去重）
        - models 为空时回退到单 model
        - 两者都为空时才使用内置默认列表
        """
        uniq: List[str] = []

        def _push(v: str) -> None:
            s = (v or "").strip()
            if s and s not in uniq:
                uniq.append(s)

        if isinstance(models, (list, tuple)):
            for m in models:
                _push(str(m))
        elif isinstance(models, str):
            # 兼容误配为字符串的场景：按逗号/分号/换行拆分
            for part in re.split(r"[,\n;]+", models):
                _push(part)

        if uniq:
            return uniq

        if (model or "").strip():
            _push(str(model))
            return uniq

        for m in DEFAULT_VISION_MODEL_CANDIDATES:
            _push(m)
        return uniq
    
    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        retry_count: int = 5,
        min_interval_seconds: float = 1.0,
        retry_backoff_base: float = 2.0,
        max_retry_wait_seconds: float = 120.0,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        models: Optional[List[str]] = None,
        usage_stats_path: Optional[str] = None,
        daily_token_limit_per_model: int = 2_000_000,
        token_switch_ratio: float = 0.75,
        enable_token_rotation: bool = True,
        doubao_max_name_len: int = 8,
    ):
        """
        初始化豆包API客户端
        
        Args:
            api_key: API Key (Bearer token)
            timeout: 请求超时时间（秒）
            retry_count: 失败/429 时最大尝试次数（含首次请求）
            min_interval_seconds: 两次请求之间的最小间隔（秒）；默认 1.0 ≈ 1 张/秒
            retry_backoff_base: 429 无 Retry-After 时指数退避底数
            max_retry_wait_seconds: 单次等待上限（秒）
            model: 单个方舟 Endpoint 模型 ID（与 models 二选一）
            api_base: API 根路径，默认 https://ark.cn-beijing.volces.com/api/v3
            models: 多模型候选列表；按日 token 用量超过 token_switch_ratio×限额时自动换下一个
            usage_stats_path: 日用量 JSON 路径（默认 src/doubao_api_usage.json）
            daily_token_limit_per_model: 单模型日 token 上限（默认 200 万）
            token_switch_ratio: 达到该比例即切换模型（默认 0.75）
            enable_token_rotation: 是否启用用量统计与轮换
            doubao_max_name_len: 豆包名称最大长度阈值（按字符计，中文通常等同汉字数）
        """
        self.api_key = api_key
        self.timeout = timeout
        self.retry_count = max(1, int(retry_count))
        self.model_candidates = self._normalize_model_candidates(models, model)
        self.model_id = self.model_candidates[0]
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.retry_backoff_base = max(1.5, float(retry_backoff_base))
        self.max_retry_wait_seconds = max(1.0, float(max_retry_wait_seconds))
        self.doubao_max_name_len = max(2, int(doubao_max_name_len))
        
        # 方舟 OpenAI 兼容接口：/chat/completions（/responses 易 404，已弃用）
        api_base = (api_base or "").strip().rstrip("/")
        if not api_base:
            api_base = "https://ark.cn-beijing.volces.com/api/v3"
        self.api_base = api_base
        self.chat_completions_url = f"{api_base}/chat/completions"
        self._rate_lock = threading.Lock()
        self._last_request_monotonic: float = 0.0
        self._last_model_used: str = ""
        # 本次会话内失效模型（404/401/403 等）黑名单，后续调用直接跳过
        self._blocked_models: set = set()
        self.usage_tracker: Optional[DoubaoTokenUsageTracker] = None
        if enable_token_rotation:
            st_path = (usage_stats_path or "").strip() or str(
                Path(__file__).resolve().parent / "doubao_api_usage.json"
            )
            self.usage_tracker = DoubaoTokenUsageTracker(
                st_path,
                int(daily_token_limit_per_model),
                float(token_switch_ratio),
            )
            self.usage_tracker.load()
        
        # # 鸟类相关的类别关键词（用于过滤）
        # self.bird_keywords = {
        #     "鸟", "鹰", "隼", "猫头鹰", "鹤", "鸠", "鸽", "燕", "喜鹊",
        #     "乌鸦", "鹭", "鹚", "鹈", "鹳", "鹮", "鹬", "鹪", "鹫",
        #     "雉", "鸡", "鸭", "鹅", "鸵", "鹇", "鹆", "鹃", "鹅",
        #     "雄", "雌", "雁", "雀", "莺", "鹀", "鸫", "鶇", "鹟",
        #     "鶚", "鶙", "鶛", "鶜", "鶝", "鶞", "鶟", "鶠", "鶡",
        #     "鹌", "鹑", "鹩", "鹨", "鹤", "鹥", "鹦", "鹧", "鹨",
        #     "雏", "雁", "雀", "小鸟", "鸟类", "禽", "鸟儿",
        #     "鹎",  # 鹎科鸟类，如白头鹎、红嘴黑鹎
        # }
        
        qps = (
            f"≤{1.0 / self.min_interval_seconds:.2f} 次/秒"
            if self.min_interval_seconds > 0
            else "无间隔限制"
        )
        rot = "开" if self.usage_tracker else "关"
        print(
            f"初始化豆包鸟类识别 API 客户端 | 模型候选数 {len(self.model_candidates)} "
            f"（用量轮换: {rot}）| "
            f"请求间隔 ≥ {self.min_interval_seconds}s ({qps}) | "
            f"接口: {self.chat_completions_url}"
        )

    def _throttle_before_request(self) -> None:
        """按 min_interval_seconds 限制调用频率，避免触发 429。"""
        if self.min_interval_seconds <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            gap = now - self._last_request_monotonic
            need = self.min_interval_seconds - gap
            if need > 0:
                time.sleep(need)
            self._last_request_monotonic = time.monotonic()

    def _wait_seconds_for_429(
        self, response: Optional[requests.Response], attempt_index: int
    ) -> float:
        """429 时优先使用 Retry-After，否则指数退避 + 少量抖动。"""
        if response is not None:
            ra = response.headers.get("Retry-After") or response.headers.get(
                "retry-after"
            )
            if ra:
                try:
                    sec = float(ra)
                    if sec > 0:
                        return min(sec, self.max_retry_wait_seconds)
                except ValueError:
                    pass
        base = self.retry_backoff_base ** attempt_index
        jitter = random.uniform(0, min(2.0, base * 0.15))
        return min(base + jitter, self.max_retry_wait_seconds)

    @staticmethod
    def _truncate_before_next_field(value: str) -> str:
        """字段值中若混入下一项标签（全角/半角分隔），截断到标签前。"""
        if not value:
            return ""
        m = re.search(
            r'\s*(?:[；;，,、]\s*)?(?:主体类型|简要说明|动物名称|英文名称|学名|识别准确率|准确率|置信度|中文名称)\s*[:：]',
            value,
        )
        if m:
            value = value[: m.start()]
        return value.strip().strip("。．. ")

    @staticmethod
    def _strip_doubao_annotations(s: str) -> str:
        """去掉豆包常带的说明括号，如（注：图中动物为蜥蜴，非鸟类）。"""
        if not s:
            return ""
        out = s
        out = re.sub(
            r"[（(]\s*注[:：].*?[）)]\s*",
            "",
            out,
            flags=re.DOTALL,
        )
        out = re.sub(
            r"[（(]\s*(?:说明|备注|提示)[:：].*?[）)]\s*",
            "",
            out,
            flags=re.DOTALL,
        )
        out = re.sub(r"[（(]\s*图中[^）)]*[）)]\s*", "", out)
        return out.strip()

    @staticmethod
    def _strip_outer_underscores(s: str) -> str:
        """去掉首尾下划线（如 __变色树蜥__、_Anolis_）；不碰词中间的 '_'。"""
        if not s:
            return ""
        return s.strip().strip("_").strip()

    @staticmethod
    def _strip_markdown_noise(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"^\*+\s*|\s*\*+$", "", s)
        s = s.replace("**", "").strip()
        return s

    def _normalize_cn_fragment(self, s: str) -> str:
        s = self._strip_doubao_annotations(s)
        s = self._strip_markdown_noise(s)
        s = self._truncate_before_next_field(s)
        m = re.search(r"_{1,}([\u4e00-\u9fff·・]{1,24})_{1,}", s)
        if m:
            s = m.group(1)
        else:
            s = self._strip_outer_underscores(s)
        s = re.sub(r"\s+", "", s)
        if re.match(r"^[\u4e00-\u9fff·・]{1,24}$", s):
            return s
        m2 = re.match(r"^[\u4e00-\u9fff·・]{2,24}", s)
        return m2.group(0) if m2 else ""

    def _normalize_en_fragment(self, s: str) -> str:
        s = self._strip_doubao_annotations(s)
        s = self._strip_markdown_noise(s)
        s = self._truncate_before_next_field(s)
        s = self._strip_outer_underscores(s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _normalize_sci_fragment(self, s: str) -> str:
        s = self._strip_doubao_annotations(s)
        s = self._strip_markdown_noise(s)
        s = self._truncate_before_next_field(s)
        s = s.replace("*", "").strip()
        s = self._strip_outer_underscores(s)
        if "(" in s:
            s = s.split("(", 1)[0].strip()
        return s

    @staticmethod
    def _parse_confidence_from_text(fragment: str) -> Optional[float]:
        m = re.search(r"(\d{1,3})\s*%", fragment)
        if not m:
            return None
        try:
            v = int(m.group(1))
            if 0 <= v <= 100:
                return v / 100.0
        except ValueError:
            pass
        return None

    @staticmethod
    def _is_unexpected_long_name(name: str, max_len: int = 24) -> bool:
        """
        判断是否为不可信名称（过长/夹带说明句/包含结构标签）。
        命中时应回退为未知，避免把大段解释文本当作物种名。
        """
        s = (name or "").strip()
        if not s:
            return False
        if len(s) > max_len:
            return True
        if re.search(r"[\n\r:：;；,，。.!?？（）()]", s):
            return True
        bad_tokens = (
            "主体类型",
            "简要说明",
            "动物名称",
            "英文名称",
            "中文名称",
            "学名",
            "识别准确率",
            "准确率",
            "置信度",
            "图中",
            "可能是",
            "无法确定",
            "看起来",
            "这只",
            "该图",
            "照片",
        )
        return any(t in s for t in bad_tokens)

    def predict(
        self,
        img_bgr: np.ndarray,
        top_k: int = 3,
        confidence_threshold: float = 0.3,
        geolocation: str = "中国",
    ) -> List[Dict]:
        """
        使用豆包API进行鸟类识别
        
        Args:
            img_bgr: OpenCV BGR 格式图像
            top_k: 返回最高的 k 个结果
            confidence_threshold: 置信度阈值
            geolocation: 地理信息，默认为"中国"
            
        Returns:
            [{"index": -1, "chinese_name": ..., "confidence": float}, ...]
            返回格式与本地模型兼容
        """
        if img_bgr is None or img_bgr.size == 0:
            return []

        # 编码图像为 Base64
        # 先缩小图像尺寸，减少数据大小
        max_size = 800
        h, w = img_bgr.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h))
        
        _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        user_prompt = (
            f"按固定字段输出，禁止解释。地点：{geolocation}\n"
            "第一行：主体类型：鸟类/其它（二选一）\n"
            "若鸟类：中文名称：...；英文名称：...；学名：...；识别准确率：0-100\n"
            "若其它：简要说明：...；识别准确率：0-100\n"
            f"名称最多{self.doubao_max_name_len}字，无法确定填“未知”。"
        )
        # 设置请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            'Content-Type': 'application/json'
        }

        # 先过滤会话内已失效模型，避免每次调用都从坏模型开始
        available_models = [m for m in self.model_candidates if m not in self._blocked_models]
        if not available_models:
            # 若都被封禁，则清空黑名单，避免完全不可用
            self._blocked_models.clear()
            available_models = list(self.model_candidates)

        # 基于日 token 阈值选择初始模型（仅在可用集合内）
        model_id = (
            self.usage_tracker.pick_model(available_models)
            if self.usage_tracker
            else available_models[0]
        )
        self._last_model_used = model_id
        model_idx = 0
        for i, m in enumerate(available_models):
            if m == model_id:
                model_idx = i
                break

        def _switch_to_next_model(
            reason: str, attempt_index: int, block_current: bool = False
        ) -> bool:
            nonlocal model_id, model_idx
            if block_current and model_id:
                self._blocked_models.add(model_id)

            # 每次切换时按最新黑名单重建可用模型池
            current_pool = [m for m in self.model_candidates if m not in self._blocked_models]
            if not current_pool:
                return False
            if len(current_pool) == 1 and current_pool[0] == model_id:
                return False

            old = model_id
            if old in current_pool:
                cur_i = current_pool.index(old)
                nxt_i = (cur_i + 1) % len(current_pool)
            else:
                nxt_i = 0
            model_id = current_pool[nxt_i]
            model_idx = nxt_i
            self._last_model_used = model_id
            print(
                f"⚠ 模型切换: {old} -> {model_id} "
                f"(原因: {reason}, attempt {attempt_index + 1}/{self.retry_count})"
            )
            return True

        for attempt in range(self.retry_count):
            data = {
                "model": model_id,
                # 关闭深度思考/推理输出，避免额外 reasoning tokens
                # （Fire/Ark OpenAI 兼容接口支持 thinking.type=disabled）
                "thinking": {"type": "disabled"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                },
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    }
                ],
            }
            self._throttle_before_request()
            try:
                response = requests.request(
                    "POST",
                    self.chat_completions_url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout,
                )

                if response.status_code == 429:
                    wait = self._wait_seconds_for_429(response, attempt)
                    if attempt < self.retry_count - 1:
                        _switch_to_next_model("429 限流/疑似 token 达上限", attempt)
                        print(
                            f"✗ 豆包API 限流(429)，{wait:.1f}s 后重试 "
                            f"({attempt + 1}/{self.retry_count})"
                        )
                        time.sleep(wait)
                        continue
                    print(
                        f"✗ 豆包API 限流(429)，已达最大重试次数 ({self.retry_count})"
                    )
                    return []

                if response.status_code == 404:
                    snippet = ""
                    try:
                        snippet = (response.text or "")[:400]
                    except Exception:
                        pass
                    print(
                        "✗ 方舟 API 404：当前请求 "
                        f"{self.chat_completions_url}\n"
                        "  请确认：① 网络可达火山引擎；② api_base 为控制台「API 访问」中的根地址"
                        "（默认 https://ark.cn-beijing.volces.com/api/v3）；\n"
                        "  ③ model 填推理接入点 ID（多为 ep- 开头），与控制台创建的一致。"
                    )
                    if snippet:
                        print(f"  响应片段: {snippet}")
                    if attempt < self.retry_count - 1:
                        switched = _switch_to_next_model(
                            "404 模型/接入点不存在或无权限", attempt, block_current=True
                        )
                        if switched:
                            time.sleep(min(2.0, 0.5 * (attempt + 1)))
                            continue
                    return []

                if response.status_code in (401, 403):
                    if attempt < self.retry_count - 1:
                        switched = _switch_to_next_model(
                            f"HTTP {response.status_code} 拒绝访问", attempt, block_current=True
                        )
                        wait = min(3.0, 0.5 * (attempt + 1))
                        if switched:
                            time.sleep(wait)
                            continue
                    print(
                        f"✗ 豆包API 访问被拒绝 (HTTP {response.status_code})，"
                        "且无可用备选模型或已达重试上限"
                    )
                    return []

                response.raise_for_status()

                response.encoding = "utf-8"
                result = response.json()
                # 默认不打印完整原始响应（其中可能包含 reasoning_content，既浪费日志也可能影响性能）

                usage = result.get("usage") or {}
                tt = usage.get("total_tokens")
                if tt is None:
                    tt = int(usage.get("prompt_tokens", 0) or 0) + int(
                        usage.get("completion_tokens", 0) or 0
                    )
                if self.usage_tracker and tt and tt > 0:
                    self.usage_tracker.add_tokens(model_id, int(tt))
                    self.usage_tracker.persist()

                return self._parse_response(result, top_k, confidence_threshold)

            except requests.exceptions.Timeout:
                print(f"✗ 请求超时 (第{attempt+1}次)")
                if attempt < self.retry_count - 1:
                    _switch_to_next_model("请求超时", attempt)
                    time.sleep(min(8.0, 2.0 * (attempt + 1)))

            except requests.exceptions.RequestException as e:
                print(f"✗ 请求失败 (第{attempt+1}次): {e}")
                if attempt < self.retry_count - 1:
                    txt = str(e)
                    if "429" in txt or "rate limit" in txt.lower():
                        _switch_to_next_model("RequestException: 限流", attempt)
                    elif "403" in txt or "401" in txt or "forbidden" in txt.lower():
                        _switch_to_next_model("RequestException: 拒绝访问", attempt)
                    time.sleep(min(6.0, 1.0 * (attempt + 1)))

            except Exception as e:
                print(f"✗ 处理异常 (第{attempt+1}次): {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(1)

        return []

    @staticmethod
    def _extract_text_from_chat_completion(result: Dict) -> str:
        """OpenAI 兼容 chat.completions 响应 → 助手纯文本。"""
        choices = result.get("choices") or []
        if not choices:
            return ""
        msg = (choices[0].get("message") or {})
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                t = block.get("type")
                if t == "text":
                    parts.append(str(block.get("text") or ""))
                elif t == "output_text":
                    parts.append(str(block.get("text") or ""))
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _extract_text_from_responses_api(result: Dict) -> str:
        """旧版 Responses 结构（input/output）兜底。"""
        output = result.get("output") or []
        for item in output:
            if item.get("type") == "message" and item.get("role") == "assistant":
                for content_item in item.get("content", []) or []:
                    if content_item.get("type") == "output_text":
                        return (content_item.get("text") or "").strip()
        return ""

    def _parse_response(
        self,
        result: Dict,
        top_k: int,
        confidence_threshold: float,
    ) -> List[Dict]:
        """
        解析豆包API响应
        
        Args:
            result: API响应数据
            top_k: 返回结果数量
            confidence_threshold: 置信度阈值
            
        Returns:
            统一格式的识别结果
        """
        results = []

        text_response = self._extract_text_from_chat_completion(result)
        if not text_response:
            text_response = self._extract_text_from_responses_api(result)

        if not text_response:
            print("✗ 无法从响应中提取文本（非 chat.completions 或 Responses 预期格式）")
            return results
        
        print(f"✓ 提取到文本响应: {text_response[:100]}...")
        
        # 解析文本响应，提取主体类型与鸟类/非鸟归档字段
        parsed_info = self._parse_bird_info(text_response)
        st = (parsed_info.get("subject_type") or "").strip()

        if parsed_info and st in ("human", "other_animal", "other"):
            # 非鸟统一归档为「其它/未分类」，不再细分人物/动物子类
            tag = "未分类"
            cn = (
                (parsed_info.get("chinese_name") or "").strip()
                or (parsed_info.get("brief_cn") or "").strip()
                or (parsed_info.get("animal_name_cn") or "").strip()
                or tag
            )
            if self._is_unexpected_long_name(cn, max_len=self.doubao_max_name_len):
                cn = "未知"
            root = SUBJECT_TYPE_TO_ARCHIVE_ROOT["other"]
            results.append({
                "index": -1,
                "subject_type": "other",
                "archive_root_cn": root,
                "archive_tag_cn": tag,
                "chinese_name": cn,
                "english_name": parsed_info.get("english_name", ""),
                "scientific_name": "",
                "confidence": float(parsed_info.get("confidence") or 0.85),
                "api_source": "doubao",
            })
            print(f"✓ 成功解析非鸟归档: {parsed_info}")
        elif parsed_info and (
            parsed_info.get("chinese_name")
            or parsed_info.get("english_name")
            or parsed_info.get("scientific_name")
        ):
            cn = (parsed_info.get("chinese_name") or "").strip()
            if self._is_unexpected_long_name(cn, max_len=self.doubao_max_name_len):
                # 鸟类中文名异常时视为无效识别，交由上层按未知处理
                print("⚠ 豆包返回的中文物种名异常，已按未知处理")
                return results
            results.append({
                "index": -1,
                "subject_type": "bird",
                "archive_root_cn": SUBJECT_TYPE_TO_ARCHIVE_ROOT["bird"],
                "archive_tag_cn": "",
                "chinese_name": cn,
                "english_name": parsed_info.get("english_name", ""),
                "scientific_name": parsed_info.get("scientific_name", ""),
                "confidence": float(parsed_info.get("confidence") or 0.9),
                "api_source": "doubao",
            })
            print(f"✓ 成功解析鸟类信息: {parsed_info}")
        else:
            print("✗ 未能解析出有效识别信息")
        
        return results
    
    def _parse_bird_info(self, text: str) -> Dict:
        """
        从豆包API的文本响应中解析鸟类信息

        豆包的响应格式示例：
        这只鸟是**红嘴黑鹎**，相关信息如下：  

        - **中文名称**：红嘴黑鹎  
        - **英文名称**：Black Bulbul  
        - **学名**：*Hypsipetes leucocephalus*
        
        或紧凑格式（注意全角分号 ；）：
        小鸊鷉；英文名称：Little Grebe；学名：Tachybaptus ruficollis；识别准确率：95%。
        中文名称：红嘴巨燕鸥、英文名称：Royal Tern、学名：Thalasseus maximus、识别准确率：80%

        其它豆包样式：
        __变色树蜥__（注：图中动物为蜥蜴，非鸟类）
        _Anolis_ 或 _Calotes versicolor_（下划线包裹、无中文）
        """
        info = {
            "subject_type": "",
            "brief_cn": "",
            "animal_name_cn": "",
            "chinese_name": "",
            "english_name": "",
            "scientific_name": "",
            "confidence": 0.9,
            "archive_root_cn": "",
            "archive_tag_cn": "",
        }
        text_raw = (text or "").strip()
        # 先去掉全局说明括号，避免「注：」里的冒号干扰「无标签首节」判断
        text = self._strip_doubao_annotations(text_raw)

        sm = re.search(r"主体类型\s*[:：]\s*([^\n；;]+)", text_raw)
        if sm:
            info["subject_type"] = self._map_subject_type_token(sm.group(1))

        # ── 1) 按全角/半角分号、换行拆成子句，再按「标签：值」提取（最稳） ──
        clauses = re.split(r"\s*[；;]\s*|\s*\n+", text)
        for raw in clauses:
            clause = raw.strip()
            if not clause:
                continue
            conf = self._parse_confidence_from_text(clause)
            if conf is not None and (
                "识别准确率" in clause
                or re.match(r"^准确率\s*[:：]", clause)
                or re.match(r"^置信度\s*[:：]", clause)
            ):
                info["confidence"] = conf
                continue

            m = re.match(r"^主体类型\s*[:：]\s*(.+)$", clause)
            if m:
                info["subject_type"] = self._map_subject_type_token(m.group(1))
                continue

            m = re.match(r"^简要说明\s*[:：]\s*(.+)$", clause)
            if m:
                v = self._truncate_before_next_field(m.group(1).strip())
                v = re.sub(r"\s+", "", v)[:80]
                if v:
                    info["brief_cn"] = v
                continue

            m = re.match(r"^动物名称\s*[:：]\s*(.+)$", clause)
            if m:
                v = self._truncate_before_next_field(m.group(1).strip())
                v = re.sub(r"\s+", "", v)[:40]
                if v:
                    info["animal_name_cn"] = v
                    if not info.get("chinese_name"):
                        info["chinese_name"] = v
                continue

            m = re.match(r"^动物类型\s*[:：]\s*(.+)$", clause)
            if m:
                v = self._truncate_before_next_field(m.group(1).strip())
                v = re.sub(r"\s+", "", v)[:20]
                if v:
                    info["archive_tag_cn"] = v
                continue

            m = re.match(r"^中文名称\s*[:：]\s*(.+)$", clause)
            if m:
                v = self._normalize_cn_fragment(m.group(1))
                if v and len(v) <= 40:
                    info["chinese_name"] = v
                continue

            m = re.match(r"^英文名称\s*[:：]\s*(.+)$", clause)
            if m:
                v = self._normalize_en_fragment(m.group(1))
                if v:
                    info["english_name"] = v
                continue

            m = re.match(r"^学名\s*[:：]\s*(.+)$", clause)
            if m:
                v = self._normalize_sci_fragment(m.group(1))
                if v:
                    info["scientific_name"] = v
                continue

            # 首节无标签：短中文或 __中文__（注：…）等
            if not info["chinese_name"] and "：" not in clause and ":" not in clause:
                v = self._normalize_cn_fragment(clause)
                if 1 <= len(v) <= 24:
                    info["chinese_name"] = v

        # ── 2) 顿号/逗号紧凑行：中文名称：A、英文名称：B ──
        if not info["chinese_name"] or not info["english_name"] or not info["scientific_name"]:
            for part in re.split(r"[、，,]", text):
                part = part.strip()
                if not part:
                    continue
                if part.startswith("中文名称") or part.startswith("中文名稱"):
                    rest = re.split(r"[:：]", part, 1)
                    if len(rest) == 2:
                        v = self._normalize_cn_fragment(rest[1])
                        if v and len(v) <= 40:
                            info["chinese_name"] = v
                elif part.startswith("英文名称"):
                    rest = re.split(r"[:：]", part, 1)
                    if len(rest) == 2:
                        v = self._normalize_en_fragment(rest[1])
                        if v:
                            info["english_name"] = v
                elif part.startswith("学名"):
                    rest = re.split(r"[:：]", part, 1)
                    if len(rest) == 2:
                        v = self._normalize_sci_fragment(rest[1])
                        if v:
                            info["scientific_name"] = v

        c2 = self._parse_confidence_from_text(text_raw)
        if c2 is not None:
            info["confidence"] = c2

        # ── 3) Markdown / 宽松正则兜底（终止符含全角；） ──
        _term = r"(?:\n|[；;]|[、，,]|$)"
        if not info["chinese_name"]:
            chinese_patterns = [
                r"\*\*中文名称\*\*[:：]\s*\*\*(.+?)\*\*",
                r"\*\*中文名称\*\*[:：]\s*(.+?)(?=" + _term + r"|英文名称|学名)",
                r"这只鸟是\*\*(.+?)\*\*",
                r"中文名称\s*[:：]\s*(.+?)(?=" + _term + r"|英文名称|学名|识别准确)",
                r"_{1,}([\u4e00-\u9fff·・]{1,24})_{1,}",
            ]
            for pattern in chinese_patterns:
                match = re.search(pattern, text, flags=re.DOTALL)
                if match:
                    cn = self._normalize_cn_fragment(match.group(1))
                    if cn and len(cn) <= 40:
                        info["chinese_name"] = cn
                        break

        if not info["english_name"]:
            english_patterns = [
                r"\*\*英文名称\*\*[:：]\s*(.+?)(?=" + _term + r"|学名|识别准确|中文名称)",
                r"英文名称\s*[:：]\s*(.+?)(?=" + _term + r"|学名|识别准确|中文名称)",
                r"_{1,}([A-Z][a-z]{2,})_{1,}",
            ]
            for pattern in english_patterns:
                match = re.search(pattern, text, flags=re.DOTALL)
                if match:
                    en = self._normalize_en_fragment(match.group(1))
                    if en:
                        info["english_name"] = en
                        break

        if not info["scientific_name"]:
            scientific_patterns = [
                r"\*\*学名\*\*[:：]\s*\*(.+?)\*",
                r"\*\*学名\*\*[:：]\s*(.+?)(?=" + _term + r"|识别准确|英文名称|中文名称)",
                r"学名\s*[:：]\s*(.+?)(?=" + _term + r"|识别准确|英文名称|中文名称)",
                r"_{1,}([A-Z][a-z]+(?:\s+[a-z][a-z]+)+)_{1,}",
            ]
            for pattern in scientific_patterns:
                match = re.search(pattern, text, flags=re.DOTALL)
                if match:
                    sn = self._normalize_sci_fragment(match.group(1))
                    if sn:
                        info["scientific_name"] = sn
                        break

        # 最终校验：去掉误入的标签/英文长串，只保留合理中文名片段
        cn = info["chinese_name"]
        if cn:
            for k in ("英文名称", "学名", "识别准确率", "准确率", "置信度"):
                if k in cn:
                    cn = cn.split(k, 1)[0].strip()
            cn = self._truncate_before_next_field(cn)
            if re.search(r"[A-Za-z]{4,}", cn):
                m = re.match(r"^([\u4e00-\u9fff·・]{1,24})", cn.strip())
                cn = m.group(1) if m else ""
            if not re.match(r"^[\u4e00-\u9fff·・]{1,24}$", cn):
                if cn and not re.search(r"[\u4e00-\u9fff]", cn):
                    cn = ""
            if len(cn) > 40:
                cn = cn[:40]
            if self._is_unexpected_long_name(cn, max_len=self.doubao_max_name_len):
                cn = ""
        info["chinese_name"] = cn

        st = (info.get("subject_type") or "").strip()
        if not st:
            if cn or info.get("english_name") or info.get("scientific_name"):
                st = "bird"
        info["subject_type"] = st

        if st == "human":
            tag = (info.get("brief_cn") or cn or "未分类").strip() or "未分类"
            info["archive_root_cn"] = SUBJECT_TYPE_TO_ARCHIVE_ROOT["human"]
            info["archive_tag_cn"] = tag
        elif st == "other_animal":
            tag = (
                info.get("archive_tag_cn")
                or info.get("animal_name_cn")
                or cn
                or "未分类"
            ).strip() or "未分类"
            info["archive_root_cn"] = SUBJECT_TYPE_TO_ARCHIVE_ROOT["other_animal"]
            info["archive_tag_cn"] = tag
        elif st == "other":
            tag = (info.get("brief_cn") or cn or "未分类").strip() or "未分类"
            info["archive_root_cn"] = SUBJECT_TYPE_TO_ARCHIVE_ROOT["other"]
            info["archive_tag_cn"] = tag
        elif st == "bird":
            info["archive_root_cn"] = SUBJECT_TYPE_TO_ARCHIVE_ROOT["bird"]

        return info

    @staticmethod
    def _map_subject_type_token(raw: str) -> str:
        v = (raw or "").strip().rstrip("。；;，,、")
        if not v:
            return ""
        if v in ("鸟类", "鸟"):
            return "bird"
        if v in ("人像", "人物", "人"):
            return "human"
        if v in ("其它动物", "其他动物", "动物"):
            return "other_animal"
        if v in ("其它", "其他"):
            return "other"
        if "鸟" in v and len(v) <= 4:
            return "bird"
        if "人像" in v or v == "人物":
            return "human"
        if "动物" in v:
            return "other_animal"
        return ""

    # def _is_bird(self, name: str) -> bool:
    #     """判断识别结果是否为鸟类"""
    #     name_lower = name.lower()
        
    #     # 中文检查
    #     for keyword in self.bird_keywords:
    #         if keyword in name:
    #             return True
        
    #     # 英文检查（常见鸟类关键词）
    #     bird_en_keywords = {
    #         "bird", "eagle", "hawk", "owl", "crane", "pigeon", "dove",
    #         "swallow", "jay", "crow", "raven", "heron", "duck", "goose",
    #         "chicken", "pheasant", "sparrow", "warbler", "robin", "finch",
    #         "thrush", "nightingale", "parrot", "macaw", "cockatoo",
    #         "penguin", "ostrich", "emu", "kiwi", "peacock", "vulture",
    #         "buzzard", "kestrel", "falcon", "stork", "pelican", "cormorant",
    #         "grebe", "puffin", "albatross", "petrel", "flycatcher",
    #         "nuthatch", "woodpecker", "kingfisher", "roller", "hoopoe",
    #         "sunbird", "hummingbird", "swiftlet", "swift",
    #     }
        
    #     for keyword in bird_en_keywords:
    #         if keyword in name_lower:
    #             return True
        
    #     return False


class HybridBirdClassifier:
    """混合鸟类分类器 - 支持本地模型和豆包API切换"""
    
    def __init__(
        self,
        doubao_config: Optional[Dict] = None,
        local_model: Optional[object] = None,
        use_local: bool = True,
        fallback_to_online: bool = True,
    ):
        """
        初始化混合分类器
        
        Args:
            doubao_config: 豆包API配置 {"api_key": ...}
            local_model: 本地模型实例（BirdSpeciesClassifier）
            use_local: 默认使用本地模型
            fallback_to_online: 本地模型失败时是否回退到在线API
        """
        self.use_local = use_local
        self.fallback_to_online = fallback_to_online
        self.local_model = local_model
        self.doubao_client = None
        
        if doubao_config:
            try:
                self.doubao_client = DoubaoBirdAPIClient(
                    api_key=doubao_config.get("api_key"),
                    timeout=doubao_config.get("timeout", 30),
                    retry_count=doubao_config.get("retry_count", 5),
                    min_interval_seconds=float(
                        doubao_config.get("min_interval_seconds", 1.0)
                    ),
                    retry_backoff_base=float(
                        doubao_config.get("retry_backoff_base", 2.0)
                    ),
                    max_retry_wait_seconds=float(
                        doubao_config.get("max_retry_wait_seconds", 120.0)
                    ),
                    model=doubao_config.get("model"),
                    models=doubao_config.get("models"),
                    usage_stats_path=doubao_config.get("usage_stats_path"),
                    daily_token_limit_per_model=int(
                        doubao_config.get("daily_token_limit_per_model", 2_000_000)
                    ),
                    token_switch_ratio=float(
                        doubao_config.get("token_switch_ratio", 0.75)
                    ),
                    enable_token_rotation=bool(
                        doubao_config.get("enable_token_rotation", True)
                    ),
                    doubao_max_name_len=int(
                        doubao_config.get(
                            "DOUBAO_MAX_NAME_LEN",
                            doubao_config.get("doubao_max_name_len", 8),
                        )
                    ),
                    api_base=doubao_config.get("api_base")
                    or doubao_config.get("base_url"),
                )
                print(f"✓ 豆包API客户端初始化成功")
            except Exception as e:
                print(f"✗ 豆包API客户端初始化失败: {e}")
                self.doubao_client = None
    
    def predict(
        self,
        img_bgr: np.ndarray,
        top_k: int = 3,
        use_online: Optional[bool] = None,
        geolocation: str = "中国",
    ) -> Tuple[List[Dict], str]:
        """
        进行物种识别
        
        Args:
            img_bgr: 图像（BGR格式）
            top_k: 返回结果数量
            use_online: 是否使用在线API（None表示使用默认设置）
            geolocation: 地理信息，默认为"中国"
            
        Returns:
            (识别结果列表, 使用的方法名称)
        """
        if use_online is None:
            use_online = not self.use_local

        results: List[Dict] = []
        method = "unknown"

        if use_online and self.doubao_client:
            try:
                results = self.doubao_client.predict(
                    img_bgr, top_k, geolocation=geolocation
                )
                mid = getattr(self.doubao_client, "_last_model_used", "") or ""
                method = f"豆包API({mid})" if mid else "豆包API"
                if results:
                    return results, method
            except Exception as e:
                print(f"✗ 豆包API识别失败: {e}")
                results = []
                if not self.fallback_to_online:
                    return [], "豆包API(失败)"

        # 本地模型：① 用户选择「本地」；② 选择「豆包」但在线无结果/无客户端时回退（避免 GUI 误存 use_local_model=false 却全程空白）
        if self.local_model:
            try_local = self.use_local
            if not try_local and use_online:
                if not self.doubao_client or not results:
                    try_local = True
            if try_local:
                try:
                    results = self.local_model.predict(img_bgr, top_k)
                    if self.use_local or not use_online:
                        method = "本地模型"
                    else:
                        method = "本地模型(豆包无可用结果，已回退)"
                    return results, method
                except Exception as e:
                    print(f"✗ 本地模型识别失败: {e}")

        # 默认用本地、但上面未跑本地时：再尝试在线回退
        if self.fallback_to_online and self.doubao_client and not use_online:
            try:
                results = self.doubao_client.predict(
                    img_bgr, top_k, geolocation=geolocation
                )
                mid = getattr(self.doubao_client, "_last_model_used", "") or ""
                method = f"豆包API({mid})(回退)" if mid else "豆包API(回退)"
                return results, method
            except Exception as e:
                print(f"✗ 豆包API识别失败(回退): {e}")

        return results, method
    
    def set_model_mode(self, use_local: bool):
        """切换模型模式"""
        self.use_local = use_local
        mode_name = "本地模型" if use_local else "豆包API"
        print(f"✓ 已切换到: {mode_name}")


# ─────────────────────────────────────────────────────────────
# 快速使用示例
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    # 测试豆包API
    print("=" * 60)
    print("豆包鸟类识别 API 测试")
    print("=" * 60)
    
    # 从命令行参数或环境变量读取凭证
    api_key = "82bb4caf-0c62-45b8-91d6-0f925a0d902a"
    
    # 初始化客户端
    client = DoubaoBirdAPIClient(api_key)
    
    # 测试图像（需要实际图像）
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                print(f"\n识别图像: {image_path}")
                results = client.predict(img, top_k=5)
                
                if results:
                    print(f"\n识别结果 (前5个):")
                    for i, r in enumerate(results, 1):
                        print(f"  {i}. {r['chinese_name']}: {r['confidence']:.2%}")
                else:
                    print("未识别到任何鸟类")
            else:
                print(f"✗ 无法读取图像: {image_path}")
        else:
            print(f"✗ 文件不存在: {image_path}")
    else:
        print("\n用法: python baidu_bird_api.py <image_path>")
        print("\n示例:")
        print("  python baidu_bird_api.py bird.jpg")
