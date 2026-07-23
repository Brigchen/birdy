# -*- coding: utf-8 -*-
"""视频裁剪与稳定：弹窗选择、算法配置、预览与处理（供 birdy_gui 调用）。"""

from __future__ import annotations

import os
import sys
import time
import shutil
import subprocess
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QGridLayout,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import QRect, Qt, QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

try:
    from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
    from PyQt5.QtMultimediaWidgets import QVideoWidget
    QT_MULTIMEDIA_AVAILABLE = True
except ImportError:
    QT_MULTIMEDIA_AVAILABLE = False

# ===== 预览视频的目标宽度（高度按比例算）=====
# 预览不需要 4K 分辨率，缩放到这个宽度可以大幅提升性能
PREVIEW_MAX_WIDTH = 960


def _video_stab_log(msg: str) -> None:
    """视频稳定弹窗相关操作一律打控制台。"""
    print(f"[Birdy 视频稳定GUI] {msg}", flush=True)


def _find_ffmpeg() -> Optional[str]:
    """查找系统中的 ffmpeg 可执行文件。"""
    try:
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            return ffmpeg_path
        project_bin = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'bin', 'ffmpeg.exe'
        )
        for path in [
            project_bin,
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            '/usr/bin/ffmpeg',
            '/usr/local/bin/ffmpeg',
        ]:
            if os.path.exists(path):
                return path
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    except Exception:
        pass
    return None


def _run_ffmpeg(cmd: list) -> tuple:
    """安全地运行 ffmpeg 命令，处理 Windows 编码问题。

    Returns:
        (returncode, stdout_str, stderr_str)
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        # 二进制模式读取，手动用 UTF-8 解码（忽略错误）
        # Windows 上 ffmpeg 输出是 UTF-8，但 subprocess 默认用 GBK 解码会报错
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')
        return result.returncode, stdout, stderr
    except Exception as e:
        return -1, "", str(e)


def _ffmpeg_trim_segment(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    audio_only: bool = False,
) -> bool:
    """使用 ffmpeg 截取视频片段（保留音频和所有声道）。

    Args:
        input_path: 输入视频路径
        output_path: 输出片段路径
        start_time: 开始时间（秒）
        end_time: 结束时间（秒），-1表示到结尾
        audio_only: 是否仅提取音频
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        _video_stab_log("⚠ 未找到 ffmpeg，音频处理不可用")
        return False

    cmd = [ffmpeg, '-y', '-i', input_path]
    if start_time > 0:
        cmd.extend(['-ss', f'{start_time:.3f}'])
    if end_time > start_time:
        duration = end_time - start_time
        cmd.extend(['-t', f'{duration:.3f}'])

    if audio_only:
        # 提取音频：统一转码为 AAC 并用 M4A 封装（比原始 .aac 更可靠）
        # 注意：output_path 应该是 .m4a 后缀
        cmd.extend([
            '-vn',
            '-c:a', 'aac', '-b:a', '192k',
            '-movflags', '+faststart',
            output_path
        ])
    else:
        cmd.extend([
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-c:a', 'aac', '-b:a', '192k',
            '-movflags', '+faststart',
            output_path
        ])

    try:
        _video_stab_log(f"执行 ffmpeg 截取: {' '.join(cmd[:8])}...")
        returncode, stdout, stderr = _run_ffmpeg(cmd)
        if returncode != 0:
            _video_stab_log(f"ffmpeg 警告: {stderr[-500:] if stderr else ''}")
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        _video_stab_log(f"ffmpeg 截取失败: {e}")
        return False


def _ffmpeg_merge_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> bool:
    """将音频合并回处理后的视频（保持所有声道）。

    Args:
        video_path: 无音频的处理后视频
        audio_path: 原始音频文件（m4a/aac/mp3等）
        output_path: 最终输出路径
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        _video_stab_log("⚠ 未找到 ffmpeg，跳过音频合并")
        return False

    cmd = [
        ffmpeg, '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '192k',
        '-map', '0:v:0', '-map', '1:a:0?',
        '-shortest',
        '-movflags', '+faststart',
        output_path
    ]

    try:
        _video_stab_log("执行 ffmpeg 音视频合并...")
        returncode, stdout, stderr = _run_ffmpeg(cmd)
        success = os.path.exists(output_path) and os.path.getsize(output_path) > 0
        if not success:
            _video_stab_log(f"ffmpeg 合并警告: {stderr[-500:] if stderr else ''}")
        return success
    except Exception as e:
        _video_stab_log(f"ffmpeg 合并失败: {e}")
        return False


def _ffmpeg_merge_audio_from_video(
    video_path: str,
    audio_source_path: str,
    output_path: str,
    audio_start: float = 0.0,
    audio_end: float = -1.0,
) -> bool:
    """从源视频中提取音频并合并到处理后的视频（保持所有声道）。

    比先提取音频文件再合并更可靠，因为直接从视频容器中读取音频流。

    Args:
        video_path: 无音频的处理后视频
        audio_source_path: 含音频的源视频（从中提取音频）
        output_path: 最终输出路径
        audio_start: 音频开始时间（秒）
        audio_end: 音频结束时间（秒），-1表示到结尾
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        _video_stab_log("⚠ 未找到 ffmpeg，跳过音频合并")
        return False

    cmd = [
        ffmpeg, '-y',
        '-i', video_path,
        '-i', audio_source_path,
    ]

    # 音频截取（如果需要）
    if audio_start > 0:
        cmd.extend(['-ss', f'{audio_start:.3f}'])
    if audio_end > audio_start and audio_end > 0:
        duration = audio_end - audio_start
        cmd.extend(['-t', f'{duration:.3f}'])

    cmd.extend([
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '192k',
        '-map', '0:v:0', '-map', '1:a:0?',
        '-shortest',
        '-movflags', '+faststart',
        output_path
    ])

    try:
        _video_stab_log("执行 ffmpeg 音视频合并...")
        returncode, stdout, stderr = _run_ffmpeg(cmd)
        success = os.path.exists(output_path) and os.path.getsize(output_path) > 0
        if not success:
            _video_stab_log(f"ffmpeg 合并警告: {stderr[-500:] if stderr else ''}")
        return success
    except Exception as e:
        _video_stab_log(f"ffmpeg 合并失败: {e}")
        return False


@dataclass
class VideoStabilizeOptions:
    """视频稳定参数配置。"""
    # 算法选择: "opencv_videostab" | "mtools_api"
    algorithm: str = "opencv_videostab"

    # 稳定模式: "standard" | "virtual_tripod" | "gimbal_follow"
    # - standard: 标准vidstab防抖（默认）
    # - virtual_tripod: 虚拟三脚架，画面钉死，只抵消微手抖
    # - gimbal_follow: 云台跟随，CSRT跟踪主体+平移跟随+vidstab去抖
    stabilizer_mode: str = "standard"

    # 时间范围（秒）
    start_time: float = 0.0  # 开始时间
    end_time: float = -1.0  # 结束时间（-1表示到结尾）

    # 空间裁剪（归一化坐标 0-1）
    crop_left: float = 0.0
    crop_right: float = 1.0
    crop_top: float = 0.0
    crop_bottom: float = 1.0

    # ===== vidstab 专业防抖参数 =====
    smoothing_window: int = 30  # 平滑窗口大小（帧），越大越稳定但延迟越高
    border_type: str = "black"  # 边缘填充: black | reflect | replicate
    border_size: int = -1  # 自动计算边缘裁剪大小（-1=根据平滑窗口自动计算）
    trim_ratio: float = 0.05  # 边缘裁剪比例（当 border_size=-1 时使用，0-0.3）
    feature_detector: str = "GFTT"  # 特征检测器类型（内置算法用）
    smoothing_radius: int = 15  # 内置算法平滑半径

    # MTools API 参数（如果可用）
    mtools_motion_profile: str = "handheld"  # handheld | sports | static
    mtools_stability_strength: float = 70.0  # 稳定强度 (0-100)

    # ===== 音频设置 =====
    keep_audio: bool = True  # 是否保留原视频音频

    # ===== 图像调整参数 =====
    brightness: float = 0.0      # 亮度 (-100 ~ +100, 0=原始)
    contrast: float = 0.0        # 对比度 (-100 ~ +100, 0=原始)
    saturation: float = 0.0      # 饱和度 (-100 ~ +100, 0=原始)
    sharpness: float = 0.0       # 锐度 (0 ~ 100, 0=不锐化)
    exposure: float = 0.0        # 曝光补偿 (-100 ~ +100, 0=原始)
    highlights: float = 0.0      # 高光恢复 (-100 ~ 0, 0=不变)
    shadows: float = 0.0         # 阴影提升 (0 ~ 100, 0=不变)
    temperature: float = 0.0     # 色温 (-100 ~ +100, 0=中性)
    tint: float = 0.0            # 色调偏移 (-100 ~ +100, 0=无偏移)

    # 输出设置
    output_fps: float = -1.0  # -1表示保持原帧率
    output_codec: str = "mp4v"  # 编码器
    output_quality: int = 95  # 质量 (1-100)


class VideoDecodeThread(QThread):
    """后台视频解码线程：解码+缩放+图像调整，UI线程定时取最新帧显示。

    架构优势：
    - 解码/缩放/图像调整 全部在后台线程，UI线程只负责显示
    - 用互斥锁保护共享帧，UI线程以固定频率取最新帧
    - 不会因为信号队列堆积导致UI卡顿，暂停/停止响应即时
    - 解码跟不上时自动丢帧，保证播放速度
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)
    ended = pyqtSignal()  # 播放到结尾

    def __init__(self, video_path: str, preview_width: int = 960, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._preview_width = preview_width

        # ===== 线程控制标志（原子操作，无需锁）=====
        self._is_running = False
        self._is_paused = True  # 初始暂停
        self._seek_request = -1  # seek 目标帧，-1 表示无
        self._playback_speed = 1.0

        # ===== 共享状态（用互斥锁保护）=====
        from threading import Lock
        self._frame_lock = Lock()
        self._shared_frame: Optional[np.ndarray] = None  # 最新帧（BGR，已缩放+调整）
        self._shared_frame_idx = 0  # 最新帧号
        self._shared_frame_new = False  # 是否有新帧未被取走

        # ===== 图像调整参数 =====
        self._brightness = 0.0
        self._contrast = 0.0
        self._saturation = 0.0
        self._sharpness = 0.0
        self._exposure = 0.0
        self._highlights = 0.0
        self._shadows = 0.0
        self._temperature = 0.0
        self._tint = 0.0
        self._adjust_changed = False  # 参数变化标记，需要重新处理当前帧

    def run(self) -> None:
        """解码主循环。"""
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            self.error.emit(f"无法打开视频: {self._video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 计算预览尺寸（等比例缩放）
        if orig_w > self._preview_width:
            scale = self._preview_width / orig_w
            preview_h = int(orig_h * scale)
        else:
            preview_h = orig_h

        frame_idx = 0
        self._is_running = True
        last_frame_bgr = None  # 最后解码的帧（用于参数变化时重新处理）

        # 播放速度基准时间
        start_time = time.time()
        start_frame = 0

        while self._is_running:
            # ===== 1. 检查 seek 请求 =====
            if self._seek_request >= 0:
                target = max(0, min(self._seek_request, total_frames - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                frame_idx = target
                self._seek_request = -1
                # 重置时间基准
                start_time = time.time()
                start_frame = target
                # seek 后读一帧
                ret, frame = cap.read()
                if ret:
                    if orig_w > self._preview_width:
                        frame = cv2.resize(frame, (self._preview_width, preview_h))
                    last_frame_bgr = frame
                    self._publish_frame(frame, frame_idx)
                    frame_idx += 1
                continue

            # ===== 2. 暂停状态 =====
            if self._is_paused:
                # 如果参数变化了，重新处理当前帧
                if self._adjust_changed and last_frame_bgr is not None:
                    self._publish_frame(last_frame_bgr, frame_idx - 1)
                    self._adjust_changed = False
                self.msleep(20)  # 短睡眠，快速响应暂停/停止
                continue

            # ===== 3. 计算目标帧（基于时间戳，保证播放速度）=====
            elapsed = time.time() - start_time
            target_frame = start_frame + int(elapsed * fps * self._playback_speed)
            target_frame = min(target_frame, total_frames - 1)

            # 如果当前帧落后于目标帧，需要追赶（解码+跳帧）
            if frame_idx < target_frame:
                frames_to_skip = target_frame - frame_idx
                if frames_to_skip > 10:
                    # 落后太多，直接 seek
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    frame_idx = target_frame
                else:
                    # 跳帧追赶
                    for _ in range(frames_to_skip - 1):
                        ret = cap.grab()
                        if not ret:
                            break
                        frame_idx += 1

                # 读取目标帧
                ret, frame = cap.read()
                if not ret or frame_idx >= total_frames:
                    self.ended.emit()
                    self._is_paused = True
                    break

                if orig_w > self._preview_width:
                    frame = cv2.resize(frame, (self._preview_width, preview_h))

                last_frame_bgr = frame
                self._publish_frame(frame, frame_idx)
                frame_idx += 1
            else:
                # 播放速度够快，等一下（短睡眠，快速响应控制）
                self.msleep(5)

        cap.release()
        self.finished.emit()

    def _publish_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> None:
        """处理并发布一帧（加锁写入共享变量）。"""
        # 应用图像调整
        adjusted = self._apply_adjustments_to_frame(frame_bgr)
        # 加锁更新共享帧
        with self._frame_lock:
            self._shared_frame = adjusted
            self._shared_frame_idx = frame_idx
            self._shared_frame_new = True

    def _apply_adjustments_to_frame(self, img_bgr: np.ndarray) -> np.ndarray:
        """在线程中应用图像调整（亮度/对比度/饱和度等）。"""
        # 快速路径：所有参数都是0，直接返回
        if (abs(self._brightness) < 0.01 and abs(self._contrast) < 0.01
                and abs(self._saturation) < 0.01 and abs(self._sharpness) < 0.01
                and abs(self._exposure) < 0.01 and abs(self._highlights) < 0.01
                and abs(self._shadows) < 0.01 and abs(self._temperature) < 0.01
                and abs(self._tint) < 0.01):
            return img_bgr

        result = img_bgr.copy()
        h, w = result.shape[:2]

        # 亮度 + 曝光补偿
        if abs(self._brightness) > 0.01 or abs(self._exposure) > 0.01:
            total_brightness = self._brightness + self._exposure
            brightness_factor = total_brightness / 100.0 * 255.0
            result = np.clip(result.astype(np.float32) + brightness_factor, 0, 255).astype(np.uint8)

        # 对比度
        if abs(self._contrast) > 0.01:
            contrast_factor = (259.0 * (self._contrast + 255.0)) / (255.0 * (259.0 - self._contrast))
            result = np.clip(contrast_factor * (result.astype(np.float32) - 128.0) + 128.0, 0, 255).astype(np.uint8)

        # 饱和度
        if abs(self._saturation) > 0.01:
            sat_factor = 1.0 + self._saturation / 100.0
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
            result_float = result.astype(np.float32)
            for c in range(3):
                result_float[:, :, c] = np.clip(
                    gray + (result_float[:, :, c] - gray) * sat_factor, 0, 255
                )
            result = result_float.astype(np.uint8)

        # 色温
        if abs(self._temperature) > 0.01:
            temp_factor = self._temperature / 100.0
            result_float = result.astype(np.float32)
            if temp_factor > 0:
                # 变暖：增加红，减少蓝
                result_float[:, :, 2] = np.clip(result_float[:, :, 2] + temp_factor * 30, 0, 255)
                result_float[:, :, 0] = np.clip(result_float[:, :, 0] - temp_factor * 15, 0, 255)
            else:
                # 变冷：增加蓝，减少红
                result_float[:, :, 0] = np.clip(result_float[:, :, 0] + abs(temp_factor) * 30, 0, 255)
                result_float[:, :, 2] = np.clip(result_float[:, :, 2] - abs(temp_factor) * 15, 0, 255)
            result = result_float.astype(np.uint8)

        # 色调
        if abs(self._tint) > 0.01:
            tint_factor = self._tint / 100.0
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + tint_factor * 30) % 180
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 锐化
        if abs(self._sharpness) > 0.01:
            sharp_amount = self._sharpness / 100.0
            kernel_size = max(3, int(sharp_amount * 5) | 1)
            blurred = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
            result = cv2.addWeighted(result, 1.0 + sharp_amount, blurred, -sharp_amount, 0)

        # 高光恢复
        if self._highlights < -0.01:
            highlight_factor = 1.0 + (self._highlights / 100.0)
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            l, a, b = cv2.split(lab)
            mask = l > 200
            l[mask] = 200 + (l[mask] - 200) * highlight_factor
            lab = cv2.merge([l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 阴影提升
        if self._shadows > 0.01:
            shadow_factor = 1.0 + (self._shadows / 100.0) * 0.5
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            l, a, b = cv2.split(lab)
            mask = l < 80
            l[mask] = l[mask] * shadow_factor
            lab = cv2.merge([l.clip(0, 255).astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return result

    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, int, bool]]:
        """获取最新帧（加锁读取，UI线程调用）。

        Returns:
            (frame_rgb, frame_index, is_new) or None
        """
        with self._frame_lock:
            if self._shared_frame is None:
                return None
            frame = self._shared_frame.copy()
            idx = self._shared_frame_idx
            is_new = self._shared_frame_new
            self._shared_frame_new = False
        return (frame, idx, is_new)

    def update_adjustments(self, adjustments: dict) -> None:
        """更新图像调整参数（UI线程调用）。"""
        self._brightness = adjustments.get('brightness', 0.0)
        self._contrast = adjustments.get('contrast', 0.0)
        self._saturation = adjustments.get('saturation', 0.0)
        self._sharpness = adjustments.get('sharpness', 0.0)
        self._exposure = adjustments.get('exposure', 0.0)
        self._highlights = adjustments.get('highlights', 0.0)
        self._shadows = adjustments.get('shadows', 0.0)
        self._temperature = adjustments.get('temperature', 0.0)
        self._tint = adjustments.get('tint', 0.0)
        self._adjust_changed = True

    def play(self) -> None:
        """开始播放。"""
        self._is_paused = False

    def pause(self) -> None:
        """暂停。"""
        self._is_paused = True

    def stop(self) -> None:
        """停止线程。"""
        self._is_running = False
        self._is_paused = False

    def seek(self, frame_idx: int) -> None:
        """跳转到指定帧。"""
        self._seek_request = frame_idx

    def set_speed(self, speed: float) -> None:
        """设置播放倍速。"""
        self._playback_speed = max(0.5, min(10.0, speed))


class VideoPreviewWidget(QWidget):
    """视频预览控件：后台解码+前台渲染，支持交互式裁剪框。

    两种播放模式：
    - "video":  后台线程解码播放（原始视频预览，性能优化）
    - "frames": 内存帧列表自绘（处理后效果预览，兼容原有逻辑）
    """

    frame_changed = pyqtSignal(int)  # 当前帧号/进度
    crop_rect_changed = pyqtSignal(float, float, float, float)  # 裁剪区域 (left, top, right, bottom)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_frame: Optional[QImage] = None
        self._original_frame: Optional[QImage] = None  # 保存原始帧（未调整）
        self._frames: List[QImage] = []
        self._current_idx = 0
        self._is_playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._next_frame)
        self._fps = 30.0
        self._preview_fps = 1.0  # 预览实际帧率（每秒1帧采样）
        self._playback_speed = 1.0  # 播放倍速 (默认1x)

        # ===== 实时视频播放模式（后台解码线程，性能优化）=====
        self._decode_thread: Optional[VideoDecodeThread] = None
        self._video_total_frames = 0
        self._video_current_frame = 0
        self._playback_mode = "frames"  # "frames" | "video"

        # 裁剪框相关属性（归一化坐标 0-1）
        self._crop_rect: Optional[Tuple[float, float, float, float]] = (0.0, 0.0, 1.0, 1.0)  # (left, top, right, bottom)
        self._is_drawing_crop = False
        self._drag_start: Optional[Tuple[float, float]] = None  # 拖拽起始点
        self._drag_end: Optional[Tuple[float, float]] = None  # 拖拽结束点

        # 图像显示区域信息
        self._image_rect: QRect = QRect()  # 图像在控件中的实际绘制区域

        # ===== 图像调整参数 =====
        self._brightness: float = 0.0       # 亮度 (-100 ~ +100)
        self._contrast: float = 0.0         # 对比度 (-100 ~ +100)
        self._saturation: float = 0.0       # 饱和度 (-100 ~ +100)
        self._sharpness: float = 0.0        # 锐度 (0 ~ 100)
        self._exposure: float = 0.0         # 曝光补偿 (-100 ~ +100)
        self._highlights: float = 0.0       # 高光恢复 (-100 ~ 0)
        self._shadows: float = 0.0          # 阴影提升 (0 ~ 100)
        self._temperature: float = 0.0      # 色温 (-100 ~ +100)
        self._tint: float = 0.0             # 色调偏移 (-100 ~ +100)

        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #555;")
        self.setMouseTracking(True)

        # UI 刷新定时器（视频模式下定时从解码线程取最新帧显示）
        self._ui_refresh_timer = QTimer(self)
        self._ui_refresh_timer.timeout.connect(self._update_from_decoder)
        self._ui_refresh_timer.setInterval(30)  # ~33fps，足够流畅且不占用UI

    def set_frames(self, frames: List[np.ndarray]) -> None:
        """设置帧列表用于预览。"""
        # 关闭实时视频（切换到帧列表模式）
        self._close_video_cap()
        self._playback_mode = "frames"

        self._frames.clear()
        for f in frames[:100]:  # 最多加载100帧预览
            qimg = self._array_to_qimage(f)
            if qimg is not None:
                self._frames.append(qimg)
        self._current_idx = 0
        if self._frames:
            self._original_frame = self._frames[0]
            self._current_frame = self._frames[0]
            self._apply_and_update()

    def set_single_frame(self, frame: np.ndarray) -> None:
        """显示单帧。"""
        # 关闭实时视频（切换到帧列表模式）
        self._close_video_cap()
        self._playback_mode = "frames"

        qimg = self._array_to_qimage(frame)
        if qimg is not None:
            self._original_frame = qimg  # 保存原始帧
            self._current_frame = qimg
            self._frames = [qimg]
            self._current_idx = 0
            self._apply_and_update()

    def set_video_path(self, path: str) -> bool:
        """设置实时播放的视频文件。

        后台线程解码 + 预览尺寸缩放 + 图像调整（全部线程内完成）
        UI 线程以固定频率取最新帧显示，永远不会卡。

        Returns:
            True 表示成功打开视频，False 表示失败
        """
        # 先关闭之前的
        self._close_video_cap()
        self._frames.clear()
        self._current_idx = 0

        # 先获取视频基本信息
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._playback_mode = "frames"
            return False

        self._video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        self._fps = fps
        self._preview_fps = fps
        self._video_current_frame = 0

        # 读取首帧用于裁剪框参考
        ret, first_frame = cap.read()
        if ret:
            self._video_current_frame = 1
            # 缩放到预览尺寸
            h, w = first_frame.shape[:2]
            if w > PREVIEW_MAX_WIDTH:
                scale = PREVIEW_MAX_WIDTH / w
                new_h = int(h * scale)
                first_frame = cv2.resize(first_frame, (PREVIEW_MAX_WIDTH, new_h))
            qimg = self._array_to_qimage(first_frame)
            if qimg is not None:
                self._original_frame = qimg
                self._current_frame = qimg
                self._apply_and_update()

        cap.release()

        # 启动后台解码线程
        self._playback_mode = "video"
        self._decode_thread = VideoDecodeThread(path, PREVIEW_MAX_WIDTH, self)
        self._decode_thread.ended.connect(self._on_decode_ended)
        self._decode_thread.error.connect(self._on_decode_error)
        self._decode_thread.set_speed(self._playback_speed)
        # 把当前图像调整参数传给解码线程
        self._sync_adjustments_to_decoder()
        self._decode_thread.start()
        # 初始暂停在第0帧
        self._decode_thread.seek(0)
        # 启动 UI 刷新定时器
        self._ui_refresh_timer.start()

        return True

    def _update_from_decoder(self) -> None:
        """UI 刷新定时器回调：从解码线程取最新帧显示。

        频率固定（~33fps），永远不会让UI线程过载，
        保证暂停/停止等UI事件即时响应。
        """
        if self._decode_thread is None:
            return

        result = self._decode_thread.get_latest_frame()
        if result is None:
            return

        frame_bgr, frame_idx, is_new = result
        if not is_new:
            return  # 没有新帧，不用更新

        self._video_current_frame = frame_idx
        qimg = self._array_to_qimage(frame_bgr)
        if qimg is not None:
            self._original_frame = qimg
            self._current_frame = qimg
            self.update()  # 只做绘制，不做图像调整（线程里已经做好了）
            self.frame_changed.emit(frame_idx)

    def _on_decode_ended(self) -> None:
        """播放到结尾。"""
        self._is_playing = False

    def _on_decode_error(self, error_msg: str) -> None:
        """解码出错。"""
        _video_stab_log(f"解码错误: {error_msg}")

    def _sync_adjustments_to_decoder(self) -> None:
        """把图像调整参数同步到解码线程。"""
        if self._decode_thread is None:
            return
        self._decode_thread.update_adjustments({
            'brightness': self._brightness,
            'contrast': self._contrast,
            'saturation': self._saturation,
            'sharpness': self._sharpness,
            'exposure': self._exposure,
            'highlights': self._highlights,
            'shadows': self._shadows,
            'temperature': self._temperature,
            'tint': self._tint,
        })

    def _close_video_cap(self) -> None:
        """关闭所有视频播放资源。"""
        # 停止 UI 刷新定时器
        self._ui_refresh_timer.stop()

        # 停止解码线程
        if self._decode_thread is not None:
            try:
                self._decode_thread.stop()
                self._decode_thread.wait(2000)  # 最多等2秒
            except Exception:
                pass
            self._decode_thread = None

        self._video_total_frames = 0
        self._video_current_frame = 0
        self._is_playing = False

    def seek_video(self, frame_idx: int) -> None:
        """跳转到指定帧（仅实时视频模式有效）。"""
        if self._playback_mode != "video":
            return
        frame_idx = max(0, min(frame_idx, max(0, self._video_total_frames - 1)))
        if self._decode_thread is not None:
            self._decode_thread.seek(frame_idx)

    def get_video_info(self) -> dict:
        """获取视频信息。"""
        return {
            "mode": self._playback_mode,
            "total_frames": self._video_total_frames if self._playback_mode == "video" else len(self._frames),
            "current_frame": self._video_current_frame if self._playback_mode == "video" else self._current_idx,
            "fps": self._fps,
        }

    @staticmethod
    def _array_to_qimage(arr: np.ndarray) -> Optional[QImage]:
        """numpy数组转QImage。"""
        try:
            if arr is None or arr.size == 0:
                return None
            if len(arr.shape) == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
            elif arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            h, w = arr.shape[:2]
            return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        except Exception:
            return None

    def set_image_adjustments(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        sharpness: float = 0.0,
        exposure: float = 0.0,
        highlights: float = 0.0,
        shadows: float = 0.0,
        temperature: float = 0.0,
        tint: float = 0.0,
    ) -> None:
        """设置图像调整参数并实时更新预览。"""
        self._brightness = brightness
        self._contrast = contrast
        self._saturation = saturation
        self._sharpness = sharpness
        self._exposure = exposure
        self._highlights = highlights
        self._shadows = shadows
        self._temperature = temperature
        self._tint = tint

        # 视频模式：参数发给解码线程，由线程在后台做调整
        if self._playback_mode == "video" and self._decode_thread is not None:
            self._sync_adjustments_to_decoder()
        else:
            # 帧列表模式：UI 线程直接计算
            self._apply_and_update()

    def get_image_adjustments(self) -> Dict[str, float]:
        """获取当前图像调整参数。"""
        return {
            "brightness": self._brightness,
            "contrast": self._contrast,
            "saturation": self._saturation,
            "sharpness": self._sharpness,
            "exposure": self._exposure,
            "highlights": self._highlights,
            "shadows": self._shadows,
            "temperature": self._temperature,
            "tint": self._tint,
        }

    def reset_image_adjustments(self) -> None:
        """重置所有图像调整为默认值。"""
        self.set_image_adjustments(
            brightness=0.0, contrast=0.0, saturation=0.0,
            sharpness=0.0, exposure=0.0, highlights=0.0,
            shadows=0.0, temperature=0.0, tint=0.0
        )

    def _apply_and_update(self) -> None:
        """应用当前调整参数到原始帧并更新显示。"""
        if self._original_frame and not self._original_frame.isNull():
            adjusted = self._apply_adjustments_to_image(self._original_frame)
            self._current_frame = adjusted
            # 如果在播放列表中，也更新当前帧
            if self._frames:
                self._frames[self._current_idx % len(self._frames)] = adjusted
            self.update()

    def _apply_adjustments_to_image(self, qimage: QImage) -> QImage:
        """对QImage应用所有图像调整参数（基于OpenCV）。"""
        try:
            # 将QImage转换为numpy数组
            width = qimage.width()
            height = qimage.height()
            ptr = qimage.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))

            # 转换为BGR格式（OpenCV标准）
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            # ===== 1. 亮度调整 =====
            if abs(self._brightness) > 0.01:
                beta = int(self._brightness * 2.55)  # 映射到 [-255, +255]
                # 使用convertScaleAbs避免类型不匹配错误
                img_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=beta)

            # ===== 2. 对比度调整 =====
            if abs(self._contrast) > 0.01:
                alpha = 1.0 + (self._contrast / 100.0)  # 因子 [0, 2]
                img_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=0)

            # ===== 3. 曝光补偿（类似亮度但更平滑）=====
            if abs(self._exposure) > 0.01:
                gamma = 1.0 + (self._exposure / 100.0)  # gamma值
                gamma = max(0.1, min(3.0, gamma))  # 限制范围
                inv_gamma = 1.0 / gamma
                table = np.array([
                    ((i / 255.0) ** inv_gamma) * 255
                    for i in range(256)
                ]).astype(np.uint8)
                img_bgr = cv2.LUT(img_bgr, table)

            # ===== 4. 饱和度调整 =====
            if abs(self._saturation) > 0.01:
                hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
                h, s, v = cv2.split(hsv)
                sat_factor = 1.0 + (self._saturation / 100.0)
                s = np.clip(s * sat_factor, 0, 255)
                hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
                img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # ===== 5. 色温调整（偏蓝/偏黄）=====
            if abs(self._temperature) > 0.01:
                temp_factor = self._temperature / 100.0
                # 增加蓝色通道，减少红色通道（冷色调）
                # 或减少蓝色通道，增加红色通道（暖色调）
                img_bgr = img_bgr.astype(np.float32)
                img_bgr[:, :, 0] = np.clip(
                    img_bgr[:, :, 0] * (1.0 - temp_factor * 0.3), 0, 255
                )  # B通道（蓝色）
                img_bgr[:, :, 2] = np.clip(
                    img_bgr[:, :, 2] * (1.0 + temp_factor * 0.3), 0, 255
                )  # R通道（红色）
                img_bgr = img_bgr.astype(np.uint8)

            # ===== 6. 色调偏移（品红/绿色）=====
            if abs(self._tint) > 0.01:
                tint_factor = self._tint / 100.0
                img_bgr = img_bgr.astype(np.float32)
                img_bgr[:, :, 1] = np.clip(
                    img_bgr[:, :, 1] * (1.0 - tint_factor * 0.15), 0, 255
                )  # G通道（绿色）
                # 品红 = R+B增加，G减少
                img_bgr[:, :, 0] = np.clip(
                    img_bgr[:, :, 0] * (1.0 + tint_factor * 0.08), 0, 255
                )  # B通道
                img_bgr[:, :, 2] = np.clip(
                    img_bgr[:, :, 2] * (1.0 + tint_factor * 0.08), 0, 255
                )  # R通道
                img_bgr = img_bgr.astype(np.uint8)

            # ===== 7. 高光恢复（降低高光）=====
            if self._highlights < -0.01:
                highlight_factor = 1.0 + (self._highlights / 100.0)  # 0~1
                lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                l, a, b = cv2.split(lab)
                # 对亮部（L > 200）进行压缩
                mask = l > 200
                l[mask] = 200 + (l[mask] - 200) * highlight_factor
                lab = cv2.merge([l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
                img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # ===== 8. 阴影提升（提亮暗部）=====
            if self._shadows > 0.01:
                shadow_factor = 1.0 + (self._shadows / 100.0) * 0.5
                lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                l, a, b = cv2.split(lab)
                # 对暗部（L < 80）进行提升
                mask = l < 80
                l[mask] = l[mask] * shadow_factor
                lab = cv2.merge([l.clip(0, 255).astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
                img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # ===== 9. 锐化处理 =====
            if self._sharpness > 0.01:
                sharp_amount = self._sharpness / 100.0  # 0~1
                kernel_size = max(3, int(sharp_amount * 5) | 1)  # 奇数核
                blurred = cv2.GaussianBlur(img_bgr, (kernel_size, kernel_size), 0)
                img_bgr = cv2.addWeighted(img_bgr, 1.0 + sharp_amount, blurred, -sharp_amount, 0)

            # 转换回RGB格式
            result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return QImage(result_rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()

        except Exception as e:
            print(f"[视频稳定GUI] 图像调整失败: {e}", flush=True)
            return qimage  # 出错时返回原图

    def set_crop_rect(self, left: float, top: float, right: float, bottom: float) -> None:
        """从外部设置裁剪区域（归一化坐标）。"""
        left = max(0.0, min(1.0, left))
        right = max(0.0, min(1.0, right))
        top = max(0.0, min(1.0, top))
        bottom = max(0.0, min(1.0, bottom))

        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top

        self._crop_rect = (left, top, right, bottom)
        self.update()

    def get_crop_rect(self) -> Tuple[float, float, float, float]:
        """获取当前裁剪区域（归一化坐标）。"""
        if self._crop_rect is None:
            return (0.0, 0.0, 1.0, 1.0)
        return self._crop_rect

    def _norm_from_widget(self, wx: int, wy: int) -> Optional[Tuple[float, float]]:
        """将控件坐标转换为归一化图像坐标（0-1）。"""
        if not self._image_rect.isValid():
            return None

        mx = wx - self._image_rect.x()
        my = wy - self._image_rect.y()

        if mx < 0 or my < 0 or mx >= self._image_rect.width() or my >= self._image_rect.height():
            return None

        nx = float(mx) / float(self._image_rect.width())
        ny = float(my) / float(self._image_rect.height())

        return (nx, ny)

    def paintEvent(self, event) -> None:
        """绘制当前帧和裁剪框。"""
        from PyQt5.QtGui import QPainter, QBrush, QColor, QPen, QFont
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # 填充背景
        painter.fillRect(self.rect(), QColor(26, 26, 26))

        if self._current_frame and not self._current_frame.isNull():
            # 缩放以适应窗口，保持比例
            img_size = self._current_frame.size()
            widget_size = self.size()
            scale = min(
                widget_size.width() / img_size.width(),
                widget_size.height() / img_size.height()
            )
            new_w = int(img_size.width() * scale)
            new_h = int(img_size.height() * scale)
            x = (widget_size.width() - new_w) // 2
            y = (widget_size.height() - new_h) // 2

            target_rect = QRect(x, y, new_w, new_h)
            self._image_rect = target_rect  # 保存图像区域用于坐标转换
            painter.drawImage(target_rect, self._current_frame)

            # 绘制边框
            pen = QPen(QColor(100, 100, 100), 1)
            painter.setPen(pen)
            painter.drawRect(target_rect)

            # ===== 绘制裁剪框 =====
            self._draw_crop_rectangle(painter, target_rect)
        else:
            self._image_rect = QRect()  # 无图像时清空

            # 显示提示文字
            painter.setPen(QColor(150, 150, 150))
            font = painter.font()
            font.setPointSize(14)
            painter.setFont(font)
            painter.drawText(
                self.rect(), Qt.AlignCenter,
                "请选择视频文件\n\n提示：可在预览图上拖拽绘制裁剪框"
            )

        painter.end()

    def _draw_crop_rectangle(self, painter: QPainter, image_rect: QRect) -> None:
        """在图像上绘制裁剪框（参考动图生成的风格）。"""
        from PyQt5.QtGui import QPen, QColor, QFont

        dw = image_rect.width()
        dh = image_rect.height()

        # 绘制已确认的裁剪框（绿色虚线）
        if self._crop_rect and len(self._crop_rect) == 4:
            left, top, right, bottom = self._crop_rect
            rx = image_rect.x() + int(round(left * dw))
            ry = image_rect.y() + int(round(top * dh))
            rw = int(round((right - left) * dw))
            rh = int(round((bottom - top) * dh))

            # 绿色虚线框
            pen_crop = QPen(QColor(50, 205, 50))  # 鲜绿色
            pen_crop.setWidth(2)
            pen_crop.setStyle(Qt.DashLine)
            painter.setPen(pen_crop)
            painter.setBrush(QColor(50, 205, 50, 20))  # 半透明绿色填充
            painter.drawRect(rx, ry, max(1, rw), max(1, rh))

            # 绘制角标（类似动图生成）
            corner_size = min(rw, rh) // 8
            corner_pen = QPen(QColor(255, 140, 0), 3)  # 橙色角标
            corner_pen.setStyle(Qt.SolidLine)
            painter.setPen(corner_pen)
            painter.setBrush(Qt.NoBrush)

            # 左上角
            painter.drawLine(rx, ry, rx + corner_size, ry)
            painter.drawLine(rx, ry, rx, ry + corner_size)
            # 右上角
            painter.drawLine(rx + rw, ry, rx + rw - corner_size, ry)
            painter.drawLine(rx + rw, ry, rx + rw, ry + corner_size)
            # 左下角
            painter.drawLine(rx, ry + rh, rx + corner_size, ry + rh)
            painter.drawLine(rx, ry + rh, rx, ry + rh - corner_size)
            # 右下角
            painter.drawLine(rx + rw, ry + rh, rx + rw - corner_size, ry + rh)
            painter.drawLine(rx + rw, ry + rh, rx + rw, ry + rh - corner_size)

            # 显示裁剪比例文字
            font = QFont("Consolas", 9)
            painter.setFont(font)
            text_pen = QPen(QColor(255, 255, 200))
            painter.setPen(text_pen)
            percent_text = f"{int((right-left)*100)}% × {int((bottom-top)*100)}%"
            painter.drawText(rx + 5, ry + 15, percent_text)

        # 绘制正在拖拽的临时裁剪框（黄色虚线）
        if self._is_drawing_crop and self._drag_start and self._drag_end:
            x0, x1 = sorted((self._drag_start[0], self._drag_end[0]))
            y0, y1 = sorted((self._drag_start[1], self._drag_end[1]))

            xa = image_rect.x() + int(round(x0 * dw))
            ya = image_rect.y() + int(round(y0 * dh))
            xb = image_rect.x() + int(round(x1 * dw))
            yb = image_rect.y() + int(round(y1 * dh))

            pen_drag = QPen(QColor(255, 220, 60))  # 黄色
            pen_drag.setWidth(2)
            pen_drag.setStyle(Qt.DashDotLine)
            painter.setPen(pen_drag)
            painter.setBrush(QColor(255, 220, 60, 30))  # 半透明黄色填充
            painter.drawRect(
                min(xa, xb),
                min(ya, yb),
                max(1, abs(xb - xa)),
                max(1, abs(yb - ya)),
            )

    def mousePressEvent(self, event) -> None:
        """鼠标按下事件：开始绘制裁剪框。"""
        if event.button() != Qt.LeftButton:
            return

        if self._current_frame is None or self._current_frame.isNull():
            return

        pn = self._norm_from_widget(event.x(), event.y())
        if pn is None:
            return

        # 开始绘制新的裁剪框
        self._is_drawing_crop = True
        self._drag_start = pn
        self._drag_end = pn
        self.update()

    def mouseMoveEvent(self, event) -> None:
        """鼠标移动事件：更新裁剪框大小。"""
        if not self._is_drawing_crop or self._drag_start is None:
            return

        pn = self._norm_from_widget(event.x(), event.y())
        if pn is None:
            return

        if event.buttons() & Qt.LeftButton:
            self._drag_end = pn
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        """鼠标释放事件：完成裁剪框绘制。"""
        if event.button() != Qt.LeftButton:
            return

        if not self._is_drawing_crop or self._drag_start is None:
            return

        pn = self._norm_from_widget(event.x(), event.y())
        if pn is None:
            self._cancel_crop_drawing()
            return

        self._drag_end = pn

        # 计算归一化坐标
        x0, x1 = sorted((self._drag_start[0], self._drag_end[0]))
        y0, y1 = sorted((self._drag_start[1], self._drag_end[1]))

        # 最小尺寸限制（避免误触）
        min_span = 0.02
        if x1 - x0 < min_span:
            c = (x0 + x1) * 0.5
            x0 = max(0.0, c - min_span * 0.5)
            x1 = min(1.0, c + min_span * 0.5)
        if y1 - y0 < min_span:
            c = (y0 + y1) * 0.5
            y0 = max(0.0, c - min_span * 0.5)
            y1 = min(1.0, c + min_span * 0.5)

        # 更新裁剪框
        self._crop_rect = (x0, y0, x1, y1)
        self._is_drawing_crop = False
        self._drag_start = None
        self._drag_end = None

        # 发出信号通知外部
        self.crop_rect_changed.emit(x0, y0, x1, y1)
        self.update()

    def _cancel_crop_drawing(self) -> None:
        """取消当前的裁剪框绘制。"""
        self._is_drawing_crop = False
        self._drag_start = None
        self._drag_end = None
        self.update()

    def _next_frame(self) -> None:
        """播放下一帧（仅帧列表模式使用，视频模式用解码线程驱动）。"""
        # 视频模式由解码线程驱动，这里不处理
        if self._playback_mode != "frames":
            return
        if not self._frames:
            return
        step = max(1, int(self._playback_speed))
        self._current_idx = (self._current_idx + step) % len(self._frames)
        self._current_frame = self._frames[self._current_idx]
        self._original_frame = self._frames[self._current_idx]
        self.frame_changed.emit(self._current_idx)
        self.update()

    def set_fps(self, fps: float) -> None:
        """设置原始视频帧率（用于计算预览采样率）。"""
        self._fps = max(1.0, min(60.0, fps))  # 限制在1-60fps范围内
        # 预览帧率：每秒1帧（因为采样策略是每秒取1帧）
        # 这样播放时 1x 速度就等于原速
        self._preview_fps = 1.0

    def set_playback_speed(self, speed: float) -> None:
        """设置播放倍速 (0.5x=半速, 1x=原速, 2x=2倍速, ...)。"""
        self._playback_speed = max(0.5, min(10.0, speed))

        if self._playback_mode == "video" and self._decode_thread is not None:
            # 视频模式：直接设置解码线程速度
            self._decode_thread.set_speed(self._playback_speed)
        elif self._is_playing:
            # 帧列表模式：重启定时器
            self.pause()
            self.play()

    def play(self) -> bool:
        """开始播放。

        Returns:
            True 表示成功开始播放，False 表示无法播放
        """
        if self._playback_mode == "video":
            # ===== 视频模式（解码线程驱动）=====
            if self._decode_thread is None or self._video_total_frames < 2:
                return False
            self._is_playing = True
            self._decode_thread.play()
            return True
        else:
            # ===== 帧列表模式 =====
            if self._frames and len(self._frames) > 1:
                self._is_playing = True
                effective_fps = max(0.5, self._preview_fps * self._playback_speed)
                interval = int(1000 / effective_fps)
                self._play_timer.start(interval)
                return True
            return False

    def pause(self) -> None:
        """暂停播放（停在当前帧）。"""
        self._is_playing = False
        self._play_timer.stop()

        if self._playback_mode == "video" and self._decode_thread is not None:
            self._decode_thread.pause()

    def stop(self) -> None:
        """停止播放并回到首帧。"""
        self._is_playing = False
        self._play_timer.stop()

        # 回到首帧
        if self._playback_mode == "video":
            # 视频模式：暂停 + seek 到第0帧
            if self._decode_thread is not None and self._video_total_frames > 0:
                self._decode_thread.pause()
                self._decode_thread.seek(0)
                self._video_current_frame = 0
                self.frame_changed.emit(0)
        else:
            # 帧列表模式：回到第0帧
            if self._frames:
                self._current_idx = 0
                self._current_frame = self._frames[0]
                self._original_frame = self._frames[0]
                self.frame_changed.emit(0)
                self.update()


class VideoPreviewWorker(QThread):
    """后台线程：加载视频预览帧（避免UI卡死）。

    采样策略: 每秒取1帧（快速加载），最多取30帧（覆盖30秒内容）。
    支持时间范围限制：只加载 start_time 到 end_time 之间的帧。
    """

    progress = pyqtSignal(int, int)  # current_frame, total_frames
    finished = pyqtSignal(list, float)  # frames_list, fps
    error = pyqtSignal(str)

    def __init__(self, video_path: str, parent=None,
                 start_time: float = 0.0, end_time: float = -1.0):
        super().__init__(parent)
        self._video_path = video_path
        self._start_time = max(0.0, start_time)
        self._end_time = end_time  # -1 表示到结尾

    def run(self) -> None:
        """在后台线程中提取预览帧（仅限时间范围内）。"""
        try:
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                self.error.emit(f"无法打开视频文件: {self._video_path}")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25.0  # 默认值

            # ★ 计算时间范围内的帧索引
            if self._end_time <= 0 or self._end_time > total_frames / fps:
                end_frame = total_frames
            else:
                end_frame = int(self._end_time * fps)

            start_frame = int(self._start_time * fps)
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))

            range_frames = end_frame - start_frame

            # 跳转到起始位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # ★ 每秒取1帧，最多取30帧（覆盖30秒）
            frame_interval = max(1, int(fps))  # 每隔fps帧取1帧 = 每秒1帧
            max_preview_frames = min(30, (range_frames // frame_interval) + 1)

            preview_frames = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret or (start_frame + frame_idx) >= end_frame:
                    break

                # 每隔 fps 帧取1帧（= 每秒1帧）
                if frame_idx % frame_interval == 0:
                    preview_frames.append(frame.copy())  # 复制一份，防止cap释放后失效

                frame_idx += 1
                if len(preview_frames) >= max_preview_frames:
                    break

                if frame_idx % (frame_interval * 5) == 0:  # 每5秒报告一次进度
                    self.progress.emit(frame_idx, total_frames)

            cap.release()

            if preview_frames:
                self.finished.emit(preview_frames, fps)
            else:
                self.error.emit("无法读取任何视频帧")

        except Exception as e:
            self.error.emit(str(e))


class VideoPreviewProcessor(QThread):
    """后台线程：执行预览处理（防抖+裁剪+调整，输出到内存）。

    与 VideoStabilizeWorker 不同：
    - 输出到内存帧列表（不写文件）
    - 采样处理（每秒1帧，最多30帧）以加快速度
    - 简化的防抖处理（可选跳过）
    """

    progress = pyqtSignal(int, int, str)  # current, total, stage_description
    finished = pyqtSignal(list, float, dict)  # frames_list, fps, info_dict
    error = pyqtSignal(str)

    def __init__(self, video_path: str, options: VideoStabilizeOptions, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._options = options

    def _apply_image_adjustments(self, frame: np.ndarray) -> np.ndarray:
        """应用图像调整参数。"""
        result = frame.copy()

        # 亮度
        if abs(self._options.brightness) > 0.01:
            result = cv2.convertScaleAbs(result, alpha=1.0, beta=self._options.brightness * 25)

        # 对比度
        if abs(self._options.contrast - 1.0) > 0.01:
            result = cv2.convertScaleAbs(result, alpha=self._options.contrast, beta=0)

        # 饱和度
        if abs(self._options.saturation - 1.0) > 0.01:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self._options.saturation, 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 锐度
        if abs(self._options.sharpness) > 0.01:
            kernel = np.array([[-1,-1,-1], [-1,9+self._options.sharpness*5,-1], [-1,-1,-1]])
            result = cv2.filter2D(result, -1, kernel)

        # 曝光补偿
        if abs(self._options.exposure) > 0.01:
            gamma = 1.0 / (1.0 + self._options.exposure)
            table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
            result = cv2.LUT(result, table)

        return result

    def run(self) -> None:
        """执行预览处理（输出到内存）。"""
        try:
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                self.error.emit(f"无法打开视频文件: {self._video_path}")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps <= 0:
                fps = 25.0

            # 计算时间范围
            start_frame = int(self._options.start_time * fps)
            if self._options.end_time > 0 and self._options.end_time < total_frames / fps:
                end_frame = int(self._options.end_time * fps)
            else:
                end_frame = total_frames

            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))
            range_frames = end_frame - start_frame

            # 计算裁剪区域
            has_crop = (
                abs(self._options.crop_left - 0.0) > 0.001 or
                abs(self._options.crop_right - 1.0) > 0.001 or
                abs(self._options.crop_top - 0.0) > 0.001 or
                abs(self._options.crop_bottom - 1.0) > 0.001
            )

            if has_crop:
                crop_x1 = int(self._options.crop_left * width)
                crop_y1 = int(self._options.crop_top * height)
                crop_x2 = int(self._options.crop_right * width)
                crop_y2 = int(self._options.crop_bottom * height)

            # 检查是否有图像调整
            has_adjustment = any([
                abs(self._options.brightness) > 0.01,
                abs(self._options.contrast - 1.0) > 0.01,
                abs(self._options.saturation - 1.0) > 0.01,
                abs(self._options.sharpness) > 0.01,
                abs(self._options.exposure) > 0.01,
            ])

            # 采样间隔：每秒取1帧，最多30帧
            frame_interval = max(1, int(fps))
            max_preview_frames = min(30, (range_frames // frame_interval) + 1)

            info = {
                'start_time': self._options.start_time,
                'end_time': self._options.end_time if self._options.end_time > 0 else -1,
                'stabilized': False,  # 预览时简化处理，暂不执行完整防抖
                'cropped': has_crop,
                'adjusted': has_adjustment,
            }

            # 跳转到起始位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            preview_frames = []
            frame_idx = 0
            processed_count = 0

            while True:
                ret, frame = cap.read()
                if not ret or (start_frame + frame_idx) >= end_frame:
                    break

                # 采样：每秒取1帧
                if frame_idx % frame_interval == 0:
                    # 应用空间裁剪
                    if has_crop:
                        processed_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    else:
                        processed_frame = frame

                    # 应用图像调整
                    if has_adjustment:
                        processed_frame = self._apply_image_adjustments(processed_frame)

                    preview_frames.append(processed_frame)
                    processed_count += 1

                    # 报告进度
                    if processed_count % 10 == 0 or len(preview_frames) >= max_preview_frames:
                        pct = int(frame_idx / range_frames * 100) if range_frames > 0 else 100
                        stage_parts = []
                        if has_crop:
                            stage_parts.append("裁剪")
                        if has_adjustment:
                            stage_parts.append("调整")
                        stage_str = "+".join(stage_parts) if stage_parts else "读取"
                        self.progress.emit(frame_idx, range_frames, stage_str)

                frame_idx += 1

                if len(preview_frames) >= max_preview_frames:
                    break

            cap.release()

            if preview_frames:
                self.finished.emit(preview_frames, fps, info)
            else:
                self.error.emit("未能提取任何预览帧")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"预览处理异常: {str(e)}")


class VideoStabilizeWorker(QThread):
    """后台线程：执行视频稳定处理。"""

    progress = pyqtSignal(int, int)  # current, total
    log_line = pyqtSignal(str)
    finished_ok = pyqtSignal(str)  # output_path
    failed = pyqtSignal(str)

    def __init__(
        self,
        input_path: str,
        output_path: str,
        options: VideoStabilizeOptions,
        parent=None,
    ):
        super().__init__(parent)
        self._input_path = input_path
        self._output_path = output_path
        self._options = options

    def run(self) -> None:
        """执行视频稳定处理。"""
        try:
            # 记录图像调整参数
            has_image_adjustments = any([
                abs(self._options.brightness) > 0.01,
                abs(self._options.contrast) > 0.01,
                abs(self._options.saturation) > 0.01,
                self._options.sharpness > 0.01,
                abs(self._options.exposure) > 0.01,
                abs(self._options.highlights) > 0.01,
                self._options.shadows > 0.01,
                abs(self._options.temperature) > 0.01,
                abs(self._options.tint) > 0.01,
            ])

            if has_image_adjustments:
                self.log_line.emit("✓ 已启用图像调整（亮度/对比度/饱和度等）")

            if self._options.algorithm == "opencv_videostab":
                self._run_opencv_videostab()
            elif self._options.algorithm == "mtools_api":
                self._run_mtools_api()
            else:
                raise ValueError(f"未知算法: {self._options.algorithm}")
        except Exception as e:
            self.failed.emit(str(e))

    def _run_opencv_videostab(self) -> None:
        """使用 vidstab 库进行稳定化（OpenCV contrib videostab 模块在多数发行版中不可用，直接走 vidstab 路径）。"""
        self.log_line.emit("正在初始化视频稳定处理...")

        cap = cv2.VideoCapture(self._input_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {self._input_path}")

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = total_frames / fps if fps > 0 else 0
        self.log_line.emit(
            f"视频信息: {width}x{height}, {fps:.1f}fps, {total_frames}帧, {duration:.1f}秒"
        )

        # 计算时间范围的帧号
        start_frame = 0
        end_frame = total_frames

        if self._options.start_time > 0:
            start_frame = int(self._options.start_time * fps)
        if self._options.end_time > 0:
            end_frame = min(int(self._options.end_time * fps), total_frames)

        self.log_line.emit(f"处理范围: 帧 {start_frame} 到 {end_frame}")

        # 直接使用 vidstab 库路径（OpenCV contrib videostab API 在多数构建中不可用）
        self._run_simple_stabilization(
            cap, fps, width, height,
            start_frame, end_frame, total_frames
        )

    def _run_simple_stabilization(
        self, cap, fps, width, height,
        start_frame, end_frame, total_frames
    ) -> None:
        """使用 vidstab 库进行专业级视频防抖 + 空间裁剪 + 音频保留。

        优化处理流程（正确顺序，高效处理）：
        1. ffmpeg 先截取指定时间片段（同时提取音频）
        2. vidstab 只对该片段做防抖（节省时间）
        3. 空间裁剪 + 图像调整
        4. ffmpeg 将处理后的视频与原音频合并（保留所有声道）
        """
        self.log_line.emit("")
        self.log_line.emit("╔══════════════════════════════════════════════════╗")
        self.log_line.emit("║  使用 vidstab 专业防抖引擎                      ║")
        self.log_line.emit("╠══════════════════════════════════════════════════╣")
        self.log_line.emit("║  处理顺序: 截取片段 → 防抖 → 裁剪调整 → 合音频║")
        self.log_line.emit("║  优化: 仅处理需要的时间段，保留原始音频声道     ║")
        self.log_line.emit("╚══════════════════════════════════════════════════╝")
        self.log_line.emit("")

        # 释放传入的cap，我们会重新打开文件
        if cap is not None and cap.isOpened():
            cap.release()

        temp_files = []  # 跟踪需要清理的临时文件

        try:
            # ===== 检查 vidstab 库 =====
            try:
                import vidstab as vidstab_module
                from vidstab import VidStab
                self.log_line.emit(f"✓ vidstab 库已加载 (版本: {vidstab_module.__version__})")
            except ImportError:
                self.log_line.emit("⚠ vidstab 未安装，使用内置简化算法...")
                cap_temp = cv2.VideoCapture(self._input_path)
                self._run_builtin_stabilization(
                    cap_temp, fps, width, height, start_frame, end_frame, total_frames
                )
                cap_temp.release()
                return

            # ===== 检查 ffmpeg =====
            has_ffmpeg = _find_ffmpeg() is not None
            if self._options.keep_audio:
                if has_ffmpeg:
                    self.log_line.emit("✓ ffmpeg 可用，将保留音频声道")
                else:
                    self.log_line.emit("⚠ ffmpeg 未找到，输出将不含音频")

            # ===== 计算空间裁剪区域 =====
            has_crop = (
                abs(self._options.crop_left - 0.0) > 0.001 or
                abs(self._options.crop_right - 1.0) > 0.001 or
                abs(self._options.crop_top - 0.0) > 0.001 or
                abs(self._options.crop_bottom - 1.0) > 0.001
            )

            if has_crop:
                crop_x1 = int(self._options.crop_left * width)
                crop_y1 = int(self._options.crop_top * height)
                crop_x2 = int(self._options.crop_right * width)
                crop_y2 = int(self._options.crop_bottom * height)
                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1
                self.log_line.emit(f"✓ 目标空间裁剪: ({crop_x1},{crop_y1}) -> ({crop_x2},{crop_y2}), 输出尺寸={crop_w}x{crop_h}")
            else:
                crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, width, height
                crop_w, crop_h = width, height

            import tempfile
            temp_dir = tempfile.mkdtemp(prefix='birdy_vidstab_')

            # ===== Step 1: ffmpeg 截取时间片段（只要设置了时间范围且ffmpeg可用就裁剪）=====
            source_path = self._input_path
            # 音频来源：有时间截取就用截取后的视频，否则用原视频
            audio_source_path = self._input_path
            audio_start_time = self._options.start_time
            audio_end_time = self._options.end_time

            need_trim = (self._options.start_time > 0.1 or (self._options.end_time > 0 and self._options.end_time < total_frames / fps - 0.1))

            if need_trim and has_ffmpeg:
                self.log_line.emit("Step 1/4: ffmpeg 截取时间片段...")
                trimmed_video_path = os.path.join(temp_dir, 'trimmed_segment.mp4')

                if _ffmpeg_trim_segment(
                    self._input_path, trimmed_video_path,
                    self._options.start_time,
                    self._options.end_time if self._options.end_time > 0 else -1
                ):
                    source_path = trimmed_video_path
                    audio_source_path = trimmed_video_path  # 音频直接从截取后的视频提取
                    audio_start_time = 0.0  # 截取后的视频从0开始
                    audio_end_time = -1.0
                    temp_files.append(trimmed_video_path)
                    self.log_line.emit(f"✓ 时间片段截取完成")

                    # 读取截取后的视频信息
                    cap_trim = cv2.VideoCapture(trimmed_video_path)
                    if cap_trim.isOpened():
                        new_fps = cap_trim.get(cv2.CAP_PROP_FPS)
                        new_width = int(cap_trim.get(cv2.CAP_PROP_FRAME_WIDTH))
                        new_height = int(cap_trim.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        new_total = int(cap_trim.get(cv2.CAP_PROP_FRAME_COUNT))
                        if new_fps > 0:
                            fps = new_fps
                        if new_width > 0 and new_height > 0:
                            width, height = new_width, new_height
                            # 重新计算裁剪坐标（基于截取后的尺寸）
                            if has_crop:
                                crop_x1 = int(self._options.crop_left * width)
                                crop_y1 = int(self._options.crop_top * height)
                                crop_x2 = int(self._options.crop_right * width)
                                crop_y2 = int(self._options.crop_bottom * height)
                                crop_w = crop_x2 - crop_x1
                                crop_h = crop_y2 - crop_y1
                        cap_trim.release()
                else:
                    self.log_line.emit("⚠ ffmpeg 截取失败，将处理完整视频")
                    need_trim = False

            # ===== Step 2: 根据模式执行防抖/跟踪处理 =====
            stabilized_video_path = os.path.join(temp_dir, 'stabilized_no_audio.mp4')
            temp_files.append(stabilized_video_path)

            mode = self._options.stabilizer_mode if hasattr(self._options, 'stabilizer_mode') else 'standard'

            if mode == "virtual_tripod":
                # ===== 虚拟三脚架模式 =====
                self.log_line.emit("Step 2/4: 虚拟三脚架模式 (virtual-tripod)...")
                self.log_line.emit("  画面钉死在首帧坐标，仅抵消微小手抖")
                self._run_virtual_tripod(source_path, stabilized_video_path, width, height, fps)
                self.log_line.emit("✓ 虚拟三脚架处理完成")

            elif mode == "gimbal_follow":
                # ===== 云台跟随模式 =====
                self.log_line.emit("Step 2/4: 云台跟随模式 (gimbal-follow)...")
                self.log_line.emit("  CSRT 跟踪主体 + 画面跟随 + vidstab 去抖")
                self._run_gimbal_follow(source_path, stabilized_video_path, width, height, fps)
                self.log_line.emit("✓ 云台跟随处理完成")

            else:
                # ===== 标准 vidstab 防抖模式 =====
                self.log_line.emit("Step 2/4: vidstab 标准防抖处理...")

                stabilizer = VidStab()

                # ===== 正确计算 vidstab 参数 =====
                smoothing_window = max(5, min(60, self._options.smoothing_window))
                border_type = self._options.border_type if self._options.border_type in ['black', 'reflect', 'replicate'] else 'black'

                # 计算 border_size：负1表示根据 trim_ratio 自动计算
                if self._options.border_size < 0:
                    # 边缘大小：平滑窗口越大，需要的边缘越多
                    border_size = max(0, int(smoothing_window * 0.5 + height * self._options.trim_ratio * 0.5))
                else:
                    border_size = max(0, self._options.border_size)

                self.log_line.emit(f"  vidstab 参数: smoothing_window={smoothing_window}, border_type={border_type}, border_size={border_size}")
                self.log_line.emit(f"  输入: {source_path}")

                try:
                    stabilizer.stabilize(
                        input_path=source_path,
                        output_path=stabilized_video_path,
                        smoothing_window=smoothing_window,
                        border_type=border_type,
                        border_size=border_size,
                        show_progress=False,
                        output_fourcc='mp4v'
                    )
                    self.log_line.emit("✓ vidstab 防抖完成")
                except Exception as e:
                    self.log_line.emit(f"⚠ vidstab 处理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # vidstab失败时，直接复制源文件
                    self.log_line.emit("回退：直接使用源视频进行裁剪调整")
                    import shutil
                    shutil.copy2(source_path, stabilized_video_path)

            # ===== Step 3: 读取防抖后视频 + 空间裁剪 + 图像调整 =====
            self.log_line.emit("Step 3/4: 应用空间裁剪与图像调整...")

            stab_cap = cv2.VideoCapture(stabilized_video_path)
            if not stab_cap.isOpened():
                # 如果打不开防抖后的文件，回退到源
                stab_cap.release()
                stab_cap = cv2.VideoCapture(source_path)
                if not stab_cap.isOpened():
                    raise IOError(f"无法打开视频进行处理")

            # 获取防抖后视频的实际尺寸（可能因为 border_size 有黑边）
            stab_width = int(stab_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            stab_height = int(stab_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            stab_fps = stab_cap.get(cv2.CAP_PROP_FPS)
            if stab_fps <= 0:
                stab_fps = fps

            # 如果防抖后尺寸变了，等比例调整裁剪区域
            if stab_width != width or stab_height != height:
                self.log_line.emit(f"  防抖后尺寸变化: {width}x{height} → {stab_width}x{stab_height}，调整裁剪坐标")
                scale_x = stab_width / width if width > 0 else 1.0
                scale_y = stab_height / height if height > 0 else 1.0
                crop_x1 = int(crop_x1 * scale_x)
                crop_y1 = int(crop_y1 * scale_y)
                crop_x2 = int(crop_x2 * scale_x)
                crop_y2 = int(crop_y2 * scale_y)
                crop_w = max(1, crop_x2 - crop_x1)
                crop_h = max(1, crop_y2 - crop_y1)

            # 输出无音频视频（临时文件，后续合并音频）
            processed_video_no_audio = os.path.join(temp_dir, 'processed_no_audio.mp4')
            temp_files.append(processed_video_no_audio)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(processed_video_no_audio, fourcc, stab_fps, (crop_w, crop_h))

            frame_count = 0
            output_count = 0

            while True:
                ret, frame = stab_cap.read()
                if not ret:
                    break

                # 空间裁剪
                if has_crop:
                    h, w = frame.shape[:2]
                    cy1 = max(0, min(crop_y1, h-1))
                    cy2 = max(cy1+1, min(crop_y2, h))
                    cx1 = max(0, min(crop_x1, w-1))
                    cx2 = max(cx1+1, min(crop_x2, w))
                    cropped_frame = frame[cy1:cy2, cx1:cx2]
                    # 如果裁剪尺寸不对，resize回去
                    if cropped_frame.shape[1] != crop_w or cropped_frame.shape[0] != crop_h:
                        cropped_frame = cv2.resize(cropped_frame, (crop_w, crop_h))
                else:
                    cropped_frame = frame

                # 图像调整
                final_frame = self._apply_image_adjustments(cropped_frame)

                out.write(final_frame)
                output_count += 1
                frame_count += 1

                if frame_count % 60 == 0:
                    self.log_line.emit(f"  裁剪+调整进度: {output_count}帧")
                    self.progress.emit(frame_count, frame_count + 100)

            stab_cap.release()
            out.release()

            self.log_line.emit(f"✓ 裁剪与图像调整完成，共 {output_count} 帧")

            # ===== Step 4: ffmpeg 合并音频 =====
            final_output = self._output_path
            merge_ok = False

            # 只有勾选了保留音频且 ffmpeg 可用时才合并
            if (self._options.keep_audio and has_ffmpeg
                    and audio_source_path and os.path.exists(audio_source_path)):
                self.log_line.emit("Step 4/4: ffmpeg 合并音频声道...")
                # 先输出到临时文件，成功后替换
                temp_final = os.path.join(temp_dir, 'final_with_audio.mp4')
                merge_ok = _ffmpeg_merge_audio_from_video(
                    processed_video_no_audio,
                    audio_source_path,
                    temp_final,
                    audio_start_time,
                    audio_end_time if audio_end_time > 0 else -1
                )
                if merge_ok:
                    import shutil
                    shutil.move(temp_final, final_output)
                    self.log_line.emit("✓ 音频合并完成，声道已保留")
                else:
                    self.log_line.emit("⚠ 音频合并失败，输出无音频版本")
                    import shutil
                    shutil.copy2(processed_video_no_audio, final_output)
            else:
                self.log_line.emit("Step 4/4: 直接输出（无音频）...")
                import shutil
                shutil.copy2(processed_video_no_audio, final_output)

            # 清理临时文件
            for f in temp_files:
                try:
                    if os.path.exists(f):
                        os.unlink(f)
                except Exception:
                    pass
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass

            self.log_line.emit("")
            self.log_line.emit("=" * 50)
            self.log_line.emit(f"✓ 视频处理完成！")
            self.log_line.emit(f"  输出文件: {final_output}")
            self.log_line.emit(f"  处理帧数: {output_count}")
            if has_crop:
                self.log_line.emit(f"  空间裁剪: {crop_w}x{crop_h}")
            if merge_ok:
                self.log_line.emit(f"  音频: 已保留原始声道")
            self.log_line.emit("=" * 50)
            self.finished_ok.emit(final_output)

        except Exception as e:
            self.log_line.emit(f"✗ 处理出错: {e}")
            traceback.print_exc()
            # 清理临时文件
            for f in temp_files:
                try:
                    if os.path.exists(f):
                        os.unlink(f)
                except Exception:
                    pass
            self.failed.emit(str(e))

    def _apply_image_adjustments(self, frame: np.ndarray) -> np.ndarray:
        """对单帧应用图像调整参数（亮度、对比度、饱和度等）。"""
        try:
            img = frame.copy()

            # ===== 1. 亮度调整 =====
            if abs(self._options.brightness) > 0.01:
                beta = int(self._options.brightness * 2.55)  # 映射到 [-255, +255]
                # 使用convertScaleAbs避免类型不匹配错误（替代cv2.add）
                img = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)

            # ===== 2. 对比度调整 =====
            if abs(self._options.contrast) > 0.01:
                alpha = 1.0 + (self._options.contrast / 100.0)  # 因子 [0, 2]
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

            # ===== 3. 曝光补偿（Gamma校正）=====
            if abs(self._options.exposure) > 0.01:
                gamma = 1.0 + (self._options.exposure / 100.0)
                gamma = max(0.1, min(3.0, gamma))
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
                img = cv2.LUT(img, table)

            # ===== 4. 饱和度调整 =====
            if abs(self._options.saturation) > 0.01:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                h, s, v = cv2.split(hsv)
                sat_factor = 1.0 + (self._options.saturation / 100.0)
                s = np.clip(s * sat_factor, 0, 255)
                hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # ===== 5. 色温调整（偏蓝/偏黄）=====
            if abs(self._options.temperature) > 0.01:
                temp_factor = self._options.temperature / 100.0
                img = img.astype(np.float32)
                img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 - temp_factor * 0.3), 0, 255)  # B通道
                img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 + temp_factor * 0.3), 0, 255)  # R通道
                img = img.astype(np.uint8)

            # ===== 6. 色调偏移（品红/绿色）=====
            if abs(self._options.tint) > 0.01:
                tint_factor = self._options.tint / 100.0
                img = img.astype(np.float32)
                img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 - tint_factor * 0.15), 0, 255)  # G通道
                img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 + tint_factor * 0.08), 0, 255)  # B通道
                img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 + tint_factor * 0.08), 0, 255)  # R通道
                img = img.astype(np.uint8)

            # ===== 7. 高光恢复（降低高光）=====
            if self._options.highlights < -0.01:
                highlight_factor = 1.0 + (self._options.highlights / 100.0)
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
                l, a, b = cv2.split(lab)
                mask = l > 200
                l[mask] = 200 + (l[mask] - 200) * highlight_factor
                lab = cv2.merge([l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # ===== 8. 阴影提升（提亮暗部）=====
            if self._options.shadows > 0.01:
                shadow_factor = 1.0 + (self._options.shadows / 100.0) * 0.5
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
                l, a, b = cv2.split(lab)
                mask = l < 80
                l[mask] = l[mask] * shadow_factor
                lab = cv2.merge([l.clip(0, 255).astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # ===== 9. 锐化处理 =====
            if self._options.sharpness > 0.01:
                sharp_amount = self._options.sharpness / 100.0
                kernel_size = max(3, int(sharp_amount * 5) | 1)
                blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
                img = cv2.addWeighted(img, 1.0 + sharp_amount, blurred, -sharp_amount, 0)

            return img

        except Exception as e:
            self.log_line.emit(f"⚠ 图像调整应用失败，使用原始帧: {e}")
            return frame

    def _run_virtual_tripod(
        self,
        input_path: str,
        output_path: str,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        """虚拟三脚架模式：以首帧为基准，将所有帧对齐到首帧坐标。

        算法（帧间累积法，稳定可靠）：
        1. 逐帧计算帧间仿射变换（光流跟踪，每帧和前一帧比，差异小更稳定）
        2. 累积帧间变换，得到首帧→当前帧的全局运动轨迹
        3. 对累积运动做指数平滑（EMA），消除帧间跳变
        4. 硬限制旋转 ±5°，平移在边缘安全区内
        5. 用约束后的逆运动做 warpAffine，把画面拉回首帧坐标
        6. 输出居中裁剪 12%，保证完全无黑边
        """
        import math

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ===== 输出参数 =====
        margin_ratio = 0.12
        margin = int(min(width, height) * margin_ratio)
        out_w = width - 2 * margin
        out_h = height - 2 * margin

        max_angle_rad = math.radians(5.0)
        max_dx = margin * 0.75
        max_dy = margin * 0.75

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

        # ===== 读取首帧 =====
        ret, ref_frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            raise IOError("无法读取第一帧")

        # 首帧直接居中裁剪输出（基准帧，不需要变换）
        ref_cropped = ref_frame[margin:margin + out_h, margin:margin + out_w].copy()
        out.write(ref_cropped)

        # ===== 光流参数 =====
        feature_params = dict(
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=20,
            blockSize=7
        )
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        prev_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

        # ===== 累积变换状态（3x3 齐次矩阵，初始为单位矩阵）=====
        # M_cum 表示 ref→curr：首帧坐标到当前帧坐标的映射
        M_cum = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)

        # EMA 平滑后的累积变换（初始和 M_cum 相同）
        sx, sy, sa = 0.0, 0.0, 0.0  # smoothed tx, ty, angle
        ss = 1.0                     # smoothed scale
        ema_alpha = 0.3              # EMA 系数：越小越平滑但延迟越大

        frame_idx = 1
        failed_frames = 0

        self.log_line.emit(f"  输出尺寸: {out_w}x{out_h} (各边裁 {margin}px, {margin_ratio*100:.0f}%)")
        self.log_line.emit(f"  旋转限制: ±5°, 平移限制: ±{max_dx:.0f}px")
        self.log_line.emit(f"  总帧数: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ===== 步骤1：检测前一帧的特征点 =====
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

            M_rel = None
            if p0 is not None and len(p0) >= 10:
                # 光流跟踪到当前帧
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, p0, None, **lk_params
                )
                if p1 is not None:
                    good_old = p0[st == 1]
                    good_new = p1[st == 1]
                    if len(good_old) >= 6:
                        # estimateAffinePartial2D(old, new)：old→new 的变换
                        # 即 new_pt = M @ old_pt，这是 prev→curr
                        M_rel_raw, inliers = cv2.estimateAffinePartial2D(
                            good_old, good_new,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=3.0
                        )
                        if M_rel_raw is not None:
                            # 帧间运动合理性检查：跳变太大说明跟丢了，丢弃
                            dx_rel = abs(M_rel_raw[0, 2])
                            dy_rel = abs(M_rel_raw[1, 2])
                            if dx_rel < 50 and dy_rel < 50:
                                M_rel = M_rel_raw.astype(np.float64)

            # ===== 步骤2：累积变换 =====
            if M_rel is not None:
                # M_rel 是 2x3，扩展为 3x3 齐次矩阵
                M_rel_h = np.array([[M_rel[0,0], M_rel[0,1], M_rel[0,2]],
                                    [M_rel[1,0], M_rel[1,1], M_rel[1,2]],
                                    [0.0,        0.0,        1.0]], dtype=np.float64)
                # 右乘：M_cum = M_rel @ M_cum，即 ref→prev 再 prev→curr
                M_cum = M_rel_h @ M_cum
            else:
                failed_frames += 1

            # ===== 步骤3：从累积矩阵提取参数 =====
            # M_cum = [[s*cosθ, -s*sinθ, tx],
            #          [s*sinθ,  s*cosθ, ty],
            #          [0,       0,      1]]
            a00, a01, tx = M_cum[0]
            a10, a11, ty = M_cum[1]
            scale = math.sqrt(a00 * a00 + a10 * a10)
            angle = math.atan2(a10, a00)

            # ===== 步骤4：硬限制（钳制）=====
            angle = max(-max_angle_rad, min(max_angle_rad, angle))
            tx = max(-max_dx, min(max_dx, tx))
            ty = max(-max_dy, min(max_dy, ty))
            scale = max(0.98, min(1.02, scale))

            # ===== 步骤5：EMA 平滑，防止帧间跳变 =====
            if frame_idx == 1:
                # 第一帧处理（视频第二帧），初始化平滑值
                sx, sy, sa, ss = tx, ty, angle, scale
            elif M_rel is not None:
                # 跟踪成功，正常做 EMA
                sx = ema_alpha * tx + (1 - ema_alpha) * sx
                sy = ema_alpha * ty + (1 - ema_alpha) * sy
                sa = ema_alpha * angle + (1 - ema_alpha) * sa
                ss = ema_alpha * scale + (1 - ema_alpha) * ss
            # 跟踪失败时保持之前的平滑值，不更新（外推）

            # ===== 步骤6：用平滑+约束后的参数构造补偿矩阵 =====
            # M_comp 是 ref→curr（和 M_cum 同方向），但用了平滑和约束后的值
            cos_a = math.cos(sa) * ss
            sin_a = math.sin(sa) * ss
            M_comp = np.array([
                [cos_a, -sin_a, sx],
                [sin_a,  cos_a, sy]
            ], dtype=np.float64)

            # ===== 重建 M_cum，防止长视频无限累积漂移 =====
            # 当运动超出 clamp 范围时，M_cum 不会"记住"被截断的部分，
            # 相机移回时能立即响应
            M_cum = np.array([
                [cos_a, -sin_a, sx],
                [sin_a,  cos_a, sy],
                [0.0,    0.0,    1.0]
            ], dtype=np.float64)

            # ===== 步骤7：构造最终 warp 矩阵（居中裁剪偏移）=====
            # 输出 (ox, oy) 映射到首帧 (ox+margin, oy+margin)
            # 首帧坐标在 curr 中为 M_comp @ (ox+margin, oy+margin, 1)
            # 展开得：curr_x = cos_a*ox - sin_a*oy + cos_a*margin - sin_a*margin + sx
            #         curr_y = sin_a*ox + cos_a*oy + sin_a*margin + cos_a*margin + sy
            M_out = np.array([
                [cos_a, -sin_a, cos_a * margin - sin_a * margin + sx],
                [sin_a,  cos_a, sin_a * margin + cos_a * margin + sy]
            ], dtype=np.float64)

            # ===== 步骤8：warpAffine 输出 =====
            stabilized = cv2.warpAffine(
                frame, M_out,
                (out_w, out_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=[0, 0, 0]
            )
            out.write(stabilized)

            prev_gray = gray.copy()
            frame_idx += 1

            if frame_idx % 30 == 0:
                self.progress.emit(frame_idx, total_frames)
                self.log_line.emit(
                    f"  虚拟三脚架进度: {frame_idx}/{total_frames} "
                    f"(dx={sx:+.1f}, dy={sy:+.1f}, θ={math.degrees(sa):+.2f}°, s={ss:.3f})"
                )

        cap.release()
        out.release()

        if failed_frames > 0:
            self.log_line.emit(f"  ⚠ {failed_frames} 帧跟踪失败，使用累积变换外推")
        self.log_line.emit(f"  ✓ 输出无黑边稳定画面 {out_w}x{out_h}")

    def _run_gimbal_follow(
        self,
        input_path: str,
        output_path: str,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        """云台跟随模式：CSRT跟踪主体 + 画面跟随平移 + vidstab平滑去抖。

        处理流程：
        1. 用 CSRT 跟踪器跟踪画面中心区域的主体（假设鸟在画面中央附近）
        2. 计算每一帧主体相对于画面中心的偏移
        3. 对偏移轨迹进行平滑（云台跟随效果）
        4. 将主体稳定在画面中心附近，叠加微小去抖
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 输出尺寸（比输入略小，用于平移跟随的边缘裁剪）
        crop_margin = int(min(width, height) * 0.15)  # 15% 裁剪余量
        out_w = width - 2 * crop_margin
        out_h = height - 2 * crop_margin

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

        # ===== 初始化 CSRT 跟踪器 =====
        tracker = cv2.TrackerCSRT_create()

        # 读取第一帧，自动选择中心区域作为初始跟踪目标
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise IOError("无法读取第一帧")

        # 初始跟踪框：画面中心 40% 区域
        init_w = int(width * 0.4)
        init_h = int(height * 0.4)
        init_x = (width - init_w) // 2
        init_y = (height - init_h) // 2
        init_bbox = (init_x, init_y, init_w, init_h)

        tracker.init(first_frame, init_bbox)

        # 存储主体中心轨迹（用于平滑）
        center_x_list = []
        center_y_list = []

        self.log_line.emit(f"  输出尺寸: {out_w}x{out_h} (裁剪余量: {crop_margin}px)")
        self.log_line.emit(f"  初始跟踪框: ({init_x},{init_y}) {init_w}x{init_h}")
        self.log_line.emit("  第一遍：跟踪主体位置...")

        # ===== 第一遍：跟踪并记录主体位置 =====
        frame_idx = 0
        track_failed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ok, bbox = tracker.update(frame)

            if ok:
                x, y, w, h = [int(v) for v in bbox]
                cx = x + w / 2.0
                cy = y + h / 2.0
            else:
                # 跟踪失败，使用上一帧位置（或中心）
                track_failed += 1
                if center_x_list:
                    cx = center_x_list[-1]
                    cy = center_y_list[-1]
                else:
                    cx = width / 2.0
                    cy = height / 2.0

            center_x_list.append(cx)
            center_y_list.append(cy)

            frame_idx += 1
            if frame_idx % 30 == 0:
                self.progress.emit(frame_idx, total_frames * 2)
                self.log_line.emit(f"  跟踪进度: {frame_idx}/{total_frames} (失败: {track_failed})")

        cap.release()

        self.log_line.emit(f"  跟踪完成，共 {len(center_x_list)} 帧")

        # ===== 对轨迹进行平滑（云台跟随效果）=====
        smoothing_window = max(5, min(60, self._options.smoothing_window))
        self.log_line.emit(f"  轨迹平滑窗口: {smoothing_window} 帧")

        # 移动平均平滑
        def smooth_trajectory(trajectory, window):
            smoothed = []
            n = len(trajectory)
            for i in range(n):
                start = max(0, i - window // 2)
                end = min(n, i + window // 2 + 1)
                avg = sum(trajectory[start:end]) / (end - start)
                smoothed.append(avg)
            return smoothed

        smooth_x = smooth_trajectory(center_x_list, smoothing_window)
        smooth_y = smooth_trajectory(center_y_list, smoothing_window)

        # ===== 第二遍：应用跟随平移 + 裁剪 + 微小去抖 =====
        self.log_line.emit("  第二遍：生成云台跟随视频...")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError("无法重新打开视频")

        frame_idx = 0

        # 额外的微小去抖：用特征点光流计算局部抖动
        prev_gray = None
        prev_offset_x = 0.0
        prev_offset_y = 0.0

        feature_params = dict(
            maxCorners=100,
            qualityLevel=0.02,
            minDistance=15,
            blockSize=5
        )
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
        )

        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= len(smooth_x):
                break

            # 目标：将平滑后的主体中心移到输出画面中心
            target_cx = out_w / 2.0
            target_cy = out_h / 2.0

            # 计算裁剪左上角坐标（使主体位于输出中心）
            crop_x = int(smooth_x[frame_idx] - out_w / 2.0)
            crop_y = int(smooth_y[frame_idx] - out_h / 2.0)

            # 限制在合法范围内
            crop_x = max(0, min(crop_x, width - out_w))
            crop_y = max(0, min(crop_y, height - out_h))

            # ===== 微小去抖：用局部特征点偏移微调 =====
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
                if p0 is not None and len(p0) >= 6:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, p0, None, **lk_params
                    )
                    if p1 is not None:
                        good_old = p0[st == 1]
                        good_new = p1[st == 1]
                        if len(good_old) >= 4:
                            # 计算平均平移偏移（仅微小抖动部分）
                            dx = np.mean(good_new[:, 0] - good_old[:, 0])
                            dy = np.mean(good_new[:, 1] - good_old[:, 1])
                            # 限制微调范围（只抵消小于5像素的微小抖动）
                            max_dither = 5.0
                            dx = max(-max_dither, min(max_dither, dx))
                            dy = max(-max_dither, min(max_dither, dy))
                            # 应用微调（反向抵消）
                            crop_x -= int(dx)
                            crop_y -= int(dy)

                            # 重新限制范围
                            crop_x = max(0, min(crop_x, width - out_w))
                            crop_y = max(0, min(crop_y, height - out_h))

            prev_gray = gray.copy()

            # 裁剪
            cropped = frame[crop_y:crop_y + out_h, crop_x:crop_x + out_w]

            # 确保尺寸正确
            if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
                cropped = cv2.resize(cropped, (out_w, out_h))

            out.write(cropped)

            frame_idx += 1
            if frame_idx % 30 == 0:
                self.progress.emit(total_frames + frame_idx, total_frames * 2)
                self.log_line.emit(f"  生成进度: {frame_idx}/{total_frames}")

        cap.release()
        out.release()

        if track_failed > 0:
            self.log_line.emit(f"  ⚠ 跟踪失败 {track_failed} 帧，使用上一位置")

    def _run_builtin_stabilization(
        self, cap, fps, width, height,
        start_frame, end_frame, total_frames
    ) -> None:
        """内置简化版稳定化：先防抖稳定，后空间裁剪（vidstab不可用时使用）。

        处理流程：
        1. 分析完整帧的运动轨迹
        2. 应用防抖变换到完整帧（获得最佳效果）
        3. 在稳定后的帧上应用空间裁剪
        4. 最后应用图像调整参数
        """
        self.log_line.emit("使用内置特征点跟踪方法进行稳定化...")
        self.log_line.emit("处理顺序: 防抖(全帧) → 裁剪 → 图像调整")

        # ===== 计算空间裁剪区域 =====
        has_crop = (
            abs(self._options.crop_left - 0.0) > 0.001 or
            abs(self._options.crop_right - 1.0) > 0.001 or
            abs(self._options.crop_top - 0.0) > 0.001 or
            abs(self._options.crop_bottom - 1.0) > 0.001
        )

        if has_crop:
            crop_x1 = int(self._options.crop_left * width)
            crop_y1 = int(self._options.crop_top * height)
            crop_x2 = int(self._options.crop_right * width)
            crop_y2 = int(self._options.crop_bottom * height)
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            self.log_line.emit(f"✓ 目标空间裁剪: ({crop_x1},{crop_y1}) -> ({crop_x2},{crop_y2}), 输出尺寸={crop_w}x{crop_h}")
        else:
            crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, width, height
            crop_w, crop_h = width, height

        # 设置输出编码
        fourcc = cv2.VideoWriter_fourcc(*self._options.output_codec)
        out = cv2.VideoWriter(self._output_path, fourcc, fps, (crop_w, crop_h))

        # 读取第一帧作为参考（使用完整帧）
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, prev_frame = cap.read()
        if not ret:
            raise IOError("无法读取视频帧")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # 特征点检测参数
        feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=7
        )

        # LK光流参数
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        transforms = []  # 存储变换矩阵
        frame_count = 0

        # 第一遍：分析运动轨迹（使用完整帧）
        self.log_line.emit("分析视频中...")
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= (end_frame - start_frame):
                break

            frame_count += 1

            # ⭐ 使用完整帧进行分析（不裁剪）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测特征点
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

            if p0 is not None and len(p0) > 10:
                # 光流跟踪
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, p0, None, **lk_params
                )

                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    if len(good_new) >= 4:
                        M, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
                        if M is not None:
                            transforms.append(M)

            prev_gray = gray.copy()

            if frame_count % 30 == 0:
                progress_pct = int(frame_count / (end_frame - start_frame) * 100)
                self.progress.emit(frame_count, end_frame - start_frame)
                self.log_line.emit(f"  分析进度: {progress_pct}%")

        self.log_line.emit(f"✓ 分析完成，收集 {len(transforms)} 个变换矩阵")

        # 平滑变换轨迹
        if transforms:
            smoothed_transforms = self._smooth_transforms(transforms)
            self.log_line.emit(f"✓ 轨迹平滑完成")
        else:
            smoothed_transforms = []
            self.log_line.emit("⚠ 未检测到足够运动，输出原始帧")

        # 第二遍：应用变换(全帧) → 裁剪 → 图像调整
        self.log_line.emit("生成视频 [防抖(全帧) → 裁剪 → 调整]...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = 0
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= (end_frame - start_frame):
                break

            frame_count += 1

            # ★ 步骤1：应用防抖变换到完整帧
            if idx < len(smoothed_transforms):
                h, w = frame.shape[:2]
                stabilized_full = cv2.warpAffine(
                    frame,
                    smoothed_transforms[idx],
                    (w, h),
                    borderMode=cv2.BORDER_REPLICATE
                )
                idx += 1
            else:
                stabilized_full = frame.copy()

            # ★ 步骤2：在稳定后的帧上应用空间裁剪
            if has_crop:
                cropped_frame = stabilized_full[crop_y1:crop_y2, crop_x1:crop_x2]
            else:
                cropped_frame = stabilized_full

            # ★ 步骤3：应用图像调整参数
            final_frame = self._apply_image_adjustments(cropped_frame)

            out.write(final_frame)

            if frame_count % 30 == 0:
                progress_pct = int(frame_count / (end_frame - start_frame) * 100)
                self.progress.emit(frame_count, end_frame - start_frame)

        cap.release()
        out.release()

        self.log_line.emit(f"✓ 视频处理完成！输出: {self._output_path}")
        self.finished_ok.emit(self._output_path)

    def _smooth_transforms(self, transforms: List) -> List:
        """平滑变换序列（简单移动平均）。"""
        if not transforms:
            return transforms

        radius = max(1, self._options.smoothing_radius // 2)
        n = len(transforms)
        smoothed = []

        for i in range(n):
            start = max(0, i - radius)
            end = min(n, i + radius + 1)

            # 平均变换矩阵
            avg_M = np.zeros_like(transforms[0])
            count = 0
            for j in range(start, end):
                avg_M += transforms[j]
                count += 1

            if count > 0:
                avg_M /= count
                smoothed.append(avg_M)
            else:
                smoothed.append(transforms[i])

        return smoothed

    def _run_mtools_api(self) -> None:
        """调用 MTools API 进行AI稳定化。"""
        self.log_line.emit("正在调用 MTools AI 视频稳定API...")

        # TODO: 实现MTools API调用
        # 这里需要根据MTools的实际API文档来实现
        # 可能的方案：
        # 1. 通过HTTP API调用MTools服务
        # 2. 使用MTools CLI工具
        # 3. 直接调用MTools Python SDK（如果有）

        # 当前先给出提示信息
        self.log_line.emit(
            "⚠ MTools API 集成待实现\n"
            "当前版本仅支持 OpenCV VideoStab\n"
            "建议：\n"
            "1. 安装MTools并配置API密钥\n"
            "2. 或使用 GyroFlow（开源免费）\n"
            "3. 或使用 Adobe Premiere Warp Stabilizer"
        )

        # Fallback到OpenCV
        self.log_line.emit("自动切换到 OpenCV VideoStab...")
        self._run_opencv_videostab()


class VideoStabilizeDialog(QDialog):
    """视频裁剪与稳定对话框。"""

    def __init__(self, parent=None, default_dir: str = "", default_output_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("视频裁剪与稳定")
        # 设置窗口属性：允许调整大小
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.setMinimumSize(900, 680)

        self._default_dir = default_dir or ""
        self._default_output_dir = default_output_dir or "./watermarked"  # 默认输出到watermarked目录
        self._video_path = ""
        self._worker: Optional[VideoStabilizeWorker] = None
        self._options = VideoStabilizeOptions()
        self._preview_loader: Optional[VideoPreviewWorker] = None
        self._preview_processor: Optional[VideoPreviewProcessor] = None

        self._setup_ui()
        self._load_state()
        # 默认以最大化方式打开
        QTimer.singleShot(0, self.showMaximized)

    def _setup_ui(self) -> None:
        """构建UI布局：左侧双列设置面板 + 右侧预览区。"""
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # ===== 左侧：控制面板（双列网格布局）=====
        from PyQt5.QtWidgets import QScrollArea

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        left_panel = QWidget()
        # 使用网格布局实现双列效果
        left_grid = QGridLayout(left_panel)
        left_grid.setSpacing(10)
        left_grid.setColumnStretch(0, 1)  # 两列等宽
        left_grid.setColumnStretch(1, 1)

        current_row = 0
        current_col = 0

        def _add_to_grid(widget):
            """将控件添加到网格中的下一个可用位置。"""
            nonlocal current_row, current_col
            left_grid.addWidget(widget, current_row, current_col)
            current_col += 1
            if current_col > 1:
                current_col = 0
                current_row += 1

        # ===== 第1行: 文件选择 (跨两列) =====
        file_group = QGroupBox("📁 文件选择")
        file_layout = QVBoxLayout(file_group)

        input_row = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("选择视频文件...")
        self.file_input.setReadOnly(True)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_video_file)
        browse_btn.setMaximumWidth(80)
        input_row.addWidget(self.file_input)
        input_row.addWidget(browse_btn)
        file_layout.addLayout(input_row)

        info_label = QLabel("支持格式: MP4, AVI, MOV, MKV, WMV")
        info_label.setStyleSheet("color: #888; font-size: 10pt;")
        file_layout.addWidget(info_label)

        left_grid.addWidget(file_group, current_row, 0, 1, 2)  # 跨2列
        current_row += 1

        # ===== 第2行: 算法选择 + 时间范围 =====
        algo_group = QGroupBox("⚙️ 稳定算法")
        algo_layout = QFormLayout(algo_group)

        self.algo_combo = QComboBox()
        self.algo_combo.addItem("OpenCV VideoStab", "opencv_videostab")
        self.algo_combo.addItem("MTools AI", "mtools_api")
        self.algo_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        algo_layout.addRow("算法:", self.algo_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("🎬 标准防抖 (standard)", "standard")
        self.mode_combo.addItem("📐 虚拟三脚架 (virtual-tripod)", "virtual_tripod")
        self.mode_combo.addItem("🦅 云台跟随 (gimbal-follow)", "gimbal_follow")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        algo_layout.addRow("模式:", self.mode_combo)

        self.algo_desc = QLabel()
        self.algo_desc.setWordWrap(True)
        self.algo_desc.setStyleSheet("color: #666; font-size: 9pt;")
        algo_layout.addRow("", self.algo_desc)
        self._update_algo_description()

        time_group = QGroupBox("⏱️ 时间范围")
        time_layout = QFormLayout(time_group)

        time_range_row = QHBoxLayout()
        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setRange(0, 9999)
        self.start_time_spin.setSuffix(" 秒")
        self.start_time_spin.setDecimals(1)
        self.start_time_spin.setValue(0)
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(-1, 9999)
        self.end_time_spin.setSuffix(" 秒")
        self.end_time_spin.setDecimals(1)
        self.end_time_spin.setValue(-1)

        # ★ 时间范围改变时自动重新加载预览
        self.start_time_spin.valueChanged.connect(self._on_time_range_changed)
        self.end_time_spin.valueChanged.connect(self._on_time_range_changed)
        time_range_row.addWidget(QLabel("从"))
        time_range_row.addWidget(self.start_time_spin)
        time_range_row.addWidget(QLabel("到"))
        time_range_row.addWidget(self.end_time_spin)
        time_layout.addRow("时段:", time_range_row)

        self.duration_label = QLabel("未选择文件")
        self.duration_label.setStyleSheet("color: #888; font-size: 9pt;")
        time_layout.addRow("时长:", self.duration_label)

        left_grid.addWidget(algo_group, current_row, 0)
        left_grid.addWidget(time_group, current_row, 1)
        current_row += 1

        # ===== 第3行: 空间裁剪 + 输出设置 =====
        space_group = QGroupBox("🔲 空间裁剪")
        space_layout = QFormLayout(space_group)

        crop_h_row = QHBoxLayout()
        self.crop_left_spin = QDoubleSpinBox()
        self.crop_left_spin.setRange(0, 1)
        self.crop_left_spin.setSingleStep(0.05)
        self.crop_left_spin.setDecimals(2)
        self.crop_left_spin.setValue(0)
        self.crop_right_spin = QDoubleSpinBox()
        self.crop_right_spin.setRange(0, 1)
        self.crop_right_spin.setSingleStep(0.05)
        self.crop_right_spin.setDecimals(2)
        self.crop_right_spin.setValue(1)
        crop_h_row.addWidget(QLabel("左"))
        crop_h_row.addWidget(self.crop_left_spin)
        crop_h_row.addWidget(QLabel("右"))
        crop_h_row.addWidget(self.crop_right_spin)

        crop_v_row = QHBoxLayout()
        self.crop_top_spin = QDoubleSpinBox()
        self.crop_top_spin.setRange(0, 1)
        self.crop_top_spin.setSingleStep(0.05)
        self.crop_top_spin.setDecimals(2)
        self.crop_top_spin.setValue(0)
        self.crop_bottom_spin = QDoubleSpinBox()
        self.crop_bottom_spin.setRange(0, 1)
        self.crop_bottom_spin.setSingleStep(0.05)
        self.crop_bottom_spin.setDecimals(2)
        self.crop_bottom_spin.setValue(1)
        crop_v_row.addWidget(QLabel("上"))
        crop_v_row.addWidget(self.crop_top_spin)
        crop_v_row.addWidget(QLabel("下"))
        crop_v_row.addWidget(self.crop_bottom_spin)

        space_layout.addRow("水平:", crop_h_row)
        space_layout.addRow("垂直:", crop_v_row)

        reset_crop_btn = QPushButton("🔄 重置")
        reset_crop_btn.setToolTip("重置裁剪区域为全画面")
        reset_crop_btn.clicked.connect(self._reset_crop_rect)
        reset_crop_btn.setMaximumWidth(70)
        space_layout.addRow("", reset_crop_btn)

        output_group = QGroupBox("💾 输出设置")
        output_layout = QFormLayout(output_group)

        out_file_row = QHBoxLayout()
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("自动生成路径...")
        self.output_input.setReadOnly(True)
        out_browse_btn = QPushButton("...")
        out_browse_btn.setMaximumWidth(35)
        out_browse_btn.clicked.connect(self._browse_output_file)
        out_file_row.addWidget(self.output_input)
        out_file_row.addWidget(out_browse_btn)
        output_layout.addRow("路径:", out_file_row)

        codec_row = QHBoxLayout()
        self.codec_combo = QComboBox()
        self.codec_combo.addItem("MP4", "mp4v")
        self.codec_combo.addItem("AVI", "XVID")
        # 切换编码器时自动更新输出文件扩展名
        self.codec_combo.currentIndexChanged.connect(self._on_codec_changed)
        codec_row.addWidget(self.codec_combo)
        codec_row.addWidget(QLabel("质量:"))
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(95)
        self.quality_spin.setMaximumWidth(60)
        codec_row.addWidget(self.quality_spin)
        output_layout.addRow("编码:", codec_row)

        left_grid.addWidget(space_group, current_row, 0)
        left_grid.addWidget(output_group, current_row, 1)
        current_row += 1

        # ===== 第4-5行: 图像调整 (跨两列，内部使用紧凑布局) =====
        adjust_group = QGroupBox("🎨 图像调整")
        adjust_grid = QGridLayout(adjust_group)
        adjust_grid.setSpacing(6)

        def _create_compact_slider(
            name: str, min_val: float, max_val: float,
            default_val: float, row_idx: int, col_idx: int
        ) -> Tuple[QSlider, QLabel]:
            label = QLabel(name)
            label.setMinimumWidth(55)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(int(min_val * 10), int(max_val * 10))
            slider.setValue(int(default_val * 10))
            value_lbl = QLabel(f"{default_val:.0f}")
            value_lbl.setMinimumWidth(30)
            value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            adjust_grid.addWidget(label, row_idx, col_idx * 3)
            adjust_grid.addWidget(slider, row_idx, col_idx * 3 + 1)
            adjust_grid.addWidget(value_lbl, row_idx, col_idx * 3 + 2)

            return slider, value_lbl

        # 第一列滑块
        self.brightness_slider, self.brightness_label = _create_compact_slider(
            "亮度", -100, 100, 0, 0, 0
        )
        self.contrast_slider, self.contrast_label = _create_compact_slider(
            "对比度", -100, 100, 0, 1, 0
        )
        self.saturation_slider, self.saturation_label = _create_compact_slider(
            "饱和度", -100, 100, 0, 2, 0
        )
        self.exposure_slider, self.exposure_label = _create_compact_slider(
            "曝光", -100, 100, 0, 3, 0
        )
        self.sharpness_slider, self.sharpness_label = _create_compact_slider(
            "锐度", 0, 100, 0, 4, 0
        )

        # 第二列滑块
        self.temperature_slider, self.temperature_label = _create_compact_slider(
            "色温", -100, 100, 0, 0, 1
        )
        self.tint_slider, self.tint_label = _create_compact_slider(
            "色调", -100, 100, 0, 1, 1
        )
        self.highlights_slider, self.highlights_label = _create_compact_slider(
            "高光", -100, 0, 0, 2, 1
        )
        self.shadows_slider, self.shadows_label = _create_compact_slider(
            "阴影", 0, 100, 0, 3, 1
        )

        # 连接信号
        for slider in [
            self.brightness_slider, self.contrast_slider, self.saturation_slider,
            self.exposure_slider, self.sharpness_slider, self.temperature_slider,
            self.tint_slider, self.highlights_slider, self.shadows_slider,
        ]:
            slider.valueChanged.connect(self._on_image_adjust_changed)

        # 重置按钮
        reset_adjust_btn = QPushButton("🔄 重置所有调整")
        reset_adjust_btn.clicked.connect(self._reset_image_adjustments)
        adjust_grid.addWidget(reset_adjust_btn, 5, 0, 1, 6)

        left_grid.addWidget(adjust_group, current_row, 0, 1, 2)  # 跨2列
        current_row += 1

        # ===== 第6行: 稳定参数 + 操作按钮 =====
        param_group = QGroupBox("🎛️ vidstab 防抖参数")
        param_layout = QFormLayout(param_group)

        self.smooth_window_spin = QSpinBox()
        self.smooth_window_spin.setRange(5, 60)
        self.smooth_window_spin.setValue(30)
        self.smooth_window_spin.setToolTip("平滑窗口大小（帧）：值越大画面越稳定，但运动滞后感越强，建议15-40")
        param_layout.addRow("平滑窗口:", self.smooth_window_spin)

        self.border_type_combo = QComboBox()
        self.border_type_combo.addItem("黑边填充 (black)", "black")
        self.border_type_combo.addItem("边缘反射 (reflect)", "reflect")
        self.border_type_combo.addItem("边缘复制 (replicate)", "replicate")
        self.border_type_combo.setToolTip("防抖变换时边缘区域的填充方式")
        param_layout.addRow("边缘填充:", self.border_type_combo)

        self.trim_ratio_spin = QDoubleSpinBox()
        self.trim_ratio_spin.setRange(0, 0.3)
        self.trim_ratio_spin.setSingleStep(0.01)
        self.trim_ratio_spin.setDecimals(2)
        self.trim_ratio_spin.setValue(0.05)
        self.trim_ratio_spin.setToolTip("自动边缘裁剪比例：用于裁剪防抖产生的黑边，0表示不自动裁剪")
        param_layout.addRow("裁剪比例:", self.trim_ratio_spin)

        self.keep_audio_check = QCheckBox("保留原始音频声道")
        self.keep_audio_check.setChecked(True)
        self.keep_audio_check.setToolTip("勾选则使用 ffmpeg 合并回原音频，保持立体声/多声道；需要系统安装 ffmpeg")
        param_layout.addRow("音频:", self.keep_audio_check)

        btn_group_widget = QWidget()
        btn_layout = QHBoxLayout(btn_group_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_btn = QPushButton("👁 预览")
        self.preview_btn.clicked.connect(self._preview_first_frame)
        self.process_btn = QPushButton("▶ 处理")
        self.process_btn.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; font-weight: bold; padding: 6px; }"
            "QPushButton:hover { background-color: #218838; }"
        )
        self.process_btn.clicked.connect(self._start_processing)
        self.cancel_btn = QPushButton("⏹ 取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_processing)

        btn_layout.addWidget(self.preview_btn)
        btn_layout.addWidget(self.process_btn)
        btn_layout.addWidget(self.cancel_btn)

        left_grid.addWidget(param_group, current_row, 0)
        left_grid.addWidget(btn_group_widget, current_row, 1)
        current_row += 1

        # ===== 第7行: 日志 (跨两列) =====
        log_group = QGroupBox("📋 处理日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 9pt;"
            "background-color: #1e1e1e; color: #d4d4d4;"
        )
        log_layout.addWidget(self.log_text)

        left_grid.addWidget(log_group, current_row, 0, 1, 2)

        # 底部弹性空间
        left_grid.setRowStretch(current_row + 1, 1)

        scroll_area.setWidget(left_panel)
        # 左侧面板占50%宽度
        main_layout.addWidget(scroll_area, 1)

        # ===== 右侧：预览区 (50%) =====
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)

        preview_title = QLabel("🎬 视频预览与裁剪选区")
        preview_title.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 4px;")
        right_layout.addWidget(preview_title)

        self.preview_widget = VideoPreviewWidget()
        right_layout.addWidget(self.preview_widget, 1)

        # 播放控制条
        control_bar = QHBoxLayout()
        self.play_btn = QPushButton("▶ 播放")
        self.play_btn.clicked.connect(self._toggle_playback)
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.clicked.connect(self._stop_playback)

        # 播放倍速选择（基于原始视频速度）
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x (慢)", "1x (原速)", "2x", "4x", "8x"])
        self.speed_combo.setCurrentIndex(1)  # 默认1x原速
        self.speed_combo.setToolTip("播放倍速：基于原视频速度")
        self.speed_combo.currentIndexChanged.connect(self._on_playback_speed_changed)
        # 设置固定宽度以保持美观
        self.speed_combo.setFixedWidth(90)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 99)
        self.frame_slider.valueChanged.connect(self._seek_frame)

        control_bar.addWidget(self.play_btn)
        control_bar.addWidget(self.stop_btn)
        control_bar.addWidget(QLabel("速度:"))
        control_bar.addWidget(self.speed_combo)
        control_bar.addWidget(self.frame_slider, 1)
        right_layout.addLayout(control_bar)

        # 预览提示信息
        hint_label = QLabel(
            "💡 提示: 在预览图上拖拽可绘制裁剪框 | "
            "调整左侧滑块可实时查看图像效果"
        )
        hint_label.setStyleSheet("color: #888; font-size: 9pt; padding: 4px;")
        hint_label.setWordWrap(True)
        right_layout.addWidget(hint_label)

        main_layout.addWidget(right_panel, 1)

        # ===== 连接信号 =====
        self._connect_crop_signals()

    def _connect_crop_signals(self) -> None:
        """连接预览控件裁剪框信号与空间范围Spinbox，实现双向同步。"""
        # 预览控件裁剪框变化 → 更新 Spinbox
        self.preview_widget.crop_rect_changed.connect(self._on_preview_crop_changed)

        # Spinbox 值变化 → 更新预览控件裁剪框
        self.crop_left_spin.valueChanged.connect(self._on_spinbox_crop_changed)
        self.crop_right_spin.valueChanged.connect(self._on_spinbox_crop_changed)
        self.crop_top_spin.valueChanged.connect(self._on_spinbox_crop_changed)
        self.crop_bottom_spin.valueChanged.connect(self._on_spinbox_crop_changed)

    @pyqtSlot(float, float, float, float)
    def _on_preview_crop_changed(self, left: float, top: float, right: float, bottom: float) -> None:
        """预览控件裁剪框变化时，更新空间范围Spinbox（阻止循环触发）。"""
        # 阻止 Spinbox 触发回传
        self.crop_left_spin.blockSignals(True)
        self.crop_right_spin.blockSignals(True)
        self.crop_top_spin.blockSignals(True)
        self.crop_bottom_spin.blockSignals(True)

        try:
            self.crop_left_spin.setValue(round(left, 2))
            self.crop_right_spin.setValue(round(right, 2))
            self.crop_top_spin.setValue(round(top, 2))
            self.crop_bottom_spin.setValue(round(bottom, 2))
            self.log_text.append(f"✓ 裁剪区域已更新: 左{left:.0%} 上{top:.0%} 右{right:.0%} 下{bottom:.0%}")
        finally:
            # 恢复信号
            self.crop_left_spin.blockSignals(False)
            self.crop_right_spin.blockSignals(False)
            self.crop_top_spin.blockSignals(False)
            self.crop_bottom_spin.blockSignals(False)

    def _on_spinbox_crop_changed(self) -> None:
        """空间范围Spinbox值变化时，更新预览控件的裁剪框显示。"""
        left = self.crop_left_spin.value()
        right = self.crop_right_spin.value()
        top = self.crop_top_spin.value()
        bottom = self.crop_bottom_spin.value()

        # 更新预览控件
        self.preview_widget.set_crop_rect(left, top, right, bottom)

    def _reset_crop_rect(self) -> None:
        """重置裁剪区域为全画面 (0-1)。"""
        self.preview_widget.set_crop_rect(0.0, 0.0, 1.0, 1.0)

        # 同步更新 Spinbox
        self.crop_left_spin.setValue(0.0)
        self.crop_right_spin.setValue(1.0)
        self.crop_top_spin.setValue(0.0)
        self.crop_bottom_spin.setValue(1.0)

        self.log_text.append("🔄 裁剪区域已重置为全画面")

    def _on_codec_changed(self) -> None:
        """编码器切换时，自动更新输出文件扩展名。"""
        current_output = self.output_input.text().strip()
        if not current_output or not self._video_path:
            return

        # 根据新编码器确定扩展名
        codec_ext_map = {
            "mp4v": ".mp4",
            "XVID": ".avi",
            "avc1": ".mkv",
        }
        current_codec = self.codec_combo.currentData()
        new_ext = codec_ext_map.get(current_codec, ".mp4")

        # 替换当前输出路径的扩展名
        output_path = Path(current_output)
        new_output = str(output_path.with_suffix(new_ext))
        self.output_input.setText(new_output)

    def _on_image_adjust_changed(self) -> None:
        """图像调整滑块值改变时，更新预览控件。"""
        # 更新标签显示
        self.brightness_label.setText(f"{self.brightness_slider.value() / 10:.0f}")
        self.contrast_label.setText(f"{self.contrast_slider.value() / 10:.0f}")
        self.saturation_label.setText(f"{self.saturation_slider.value() / 10:.0f}")
        self.exposure_label.setText(f"{self.exposure_slider.value() / 10:.0f}")
        self.sharpness_label.setText(f"{self.sharpness_slider.value() / 10:.0f}")
        self.temperature_label.setText(f"{self.temperature_slider.value() / 10:.0f}")
        self.tint_label.setText(f"{self.tint_slider.value() / 10:.0f}")
        self.highlights_label.setText(f"{self.highlights_slider.value() / 10:.0f}")
        self.shadows_label.setText(f"{self.shadows_slider.value() / 10:.0f}")

        # 应用到预览控件
        self.preview_widget.set_image_adjustments(
            brightness=self.brightness_slider.value() / 10.0,
            contrast=self.contrast_slider.value() / 10.0,
            saturation=self.saturation_slider.value() / 10.0,
            exposure=self.exposure_slider.value() / 10.0,
            sharpness=self.sharpness_slider.value() / 10.0,
            temperature=self.temperature_slider.value() / 10.0,
            tint=self.tint_slider.value() / 10.0,
            highlights=self.highlights_slider.value() / 10.0,
            shadows=self.shadows_slider.value() / 10.0,
        )

    def _reset_image_adjustments(self) -> None:
        """重置所有图像调整为默认值。"""
        # 重置所有滑块
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.saturation_slider.setValue(0)
        self.exposure_slider.setValue(0)
        self.sharpness_slider.setValue(0)
        self.temperature_slider.setValue(0)
        self.tint_slider.setValue(0)
        self.highlights_slider.setValue(0)
        self.shadows_slider.setValue(0)

        # 重置预览控件的调整参数
        self.preview_widget.reset_image_adjustments()

        self.log_text.append("🔄 所有图像调整已重置")

    def _on_algorithm_changed(self, index: int) -> None:
        """算法切换时更新UI。"""
        algo = self.algo_combo.currentData()
        self._options.algorithm = algo
        self._update_algo_description()

        # vidstab 参数在使用 opencv_videostab 时启用（该模式实际使用 vidstab 库）
        is_vidstab = (algo == "opencv_videostab")
        self.smooth_window_spin.setEnabled(is_vidstab)
        self.trim_ratio_spin.setEnabled(is_vidstab)
        self.border_type_combo.setEnabled(is_vidstab)
        self.mode_combo.setEnabled(is_vidstab)

    def _on_mode_changed(self, index: int) -> None:
        """模式切换时更新UI和参数说明。"""
        mode = self.mode_combo.currentData()
        self._update_algo_description()

        # 根据模式启用/禁用相关参数
        if mode == "virtual_tripod":
            # 三脚架模式：固定参数，禁用手动调节
            self.smooth_window_spin.setEnabled(False)
            self.trim_ratio_spin.setEnabled(False)
            self.border_type_combo.setEnabled(False)
        elif mode == "gimbal_follow":
            # 云台跟随模式：需要手动选择跟踪目标，平滑参数可调
            self.smooth_window_spin.setEnabled(True)
            self.trim_ratio_spin.setEnabled(True)
            self.border_type_combo.setEnabled(True)
        else:
            # 标准模式：所有参数可调
            self.smooth_window_spin.setEnabled(True)
            self.trim_ratio_spin.setEnabled(True)
            self.border_type_combo.setEnabled(True)

    def _update_algo_description(self) -> None:
        """更新算法说明文字。"""
        algo = self.algo_combo.currentData()
        mode = self.mode_combo.currentData() if hasattr(self, 'mode_combo') else "standard"

        mode_descriptions = {
            "standard": (
                "• 标准 vidstab 防抖\n"
                "• 平滑窗口可调，适合一般手持拍摄\n"
            ),
            "virtual_tripod": (
                "📐 虚拟三脚架模式\n"
                "• 画面钉死在同一坐标\n"
                "• 仅抵消微小手抖平移/旋转\n"
                "• 适合固定机位拍摄\n"
            ),
            "gimbal_follow": (
                "🦅 云台跟随模式\n"
                "• CSRT 跟踪 + 画面跟随主体\n"
                "• 叠加 vidstab 平滑去抖\n"
                "• 适合拍摄飞鸟等运动主体\n"
            ),
        }

        self.algo_desc.setText(mode_descriptions.get(mode, ""))

    def _browse_video_file(self) -> None:
        """浏览选择视频文件。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            self._default_dir,
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;所有文件 (*)"
        )
        if file_path:
            self._set_video_file(file_path)

    def _set_video_file(self, path: str) -> None:
        """设置输入视频文件（实时播放模式，打开即可播放）。"""
        self._video_path = path
        self.file_input.setText(path)

        # 先显示"正在加载"提示
        self.duration_label.setText("⏳ 正在加载视频...")
        self.duration_label.setStyleSheet("color: #ff9800; font-size: 10pt; font-weight: bold;")

        # ★ 使用实时视频模式：打开即可播放，无需预加载帧
        ok = self.preview_widget.set_video_path(path)
        if ok:
            info = self.preview_widget.get_video_info()
            total_frames = info["total_frames"]
            fps = info["fps"]

            # 再获取一次精确的宽高（从视频元数据，比从帧取准）
            cap_tmp = cv2.VideoCapture(path)
            width = height = 0
            if cap_tmp.isOpened():
                width = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_tmp.release()

            duration = total_frames / fps if fps > 0 else 0

            self.duration_label.setText(
                f"{duration:.1f}秒 ({total_frames}帧, {width}x{height}, {fps:.1f}fps) - 可直接播放"
            )
            self.duration_label.setStyleSheet("color: #4CAF50; font-size: 9pt; font-weight: bold;")
            self.end_time_spin.setMaximum(duration)
            self.end_time_spin.setValue(-1)

            # 自动生成输出路径（默认输出到watermarked目录）
            input_path = Path(path)
            codec_ext_map = {
                "mp4v": ".mp4",
                "XVID": ".avi",
                "avc1": ".mkv",
            }
            current_codec = self.codec_combo.currentData() if hasattr(self, 'codec_combo') else "mp4v"
            output_ext = codec_ext_map.get(current_codec, ".mp4")
            output_name = f"{input_path.stem}_stabilized{output_ext}"
            output_dir = Path(self._default_output_dir)
            output_path = str(output_dir / output_name)
            self.output_input.setText(output_path)

            _video_stab_log(f"已加载视频: {path}")
            _video_stab_log(f"提示: 点击 ▶ 播放 可预览原始视频，可在预览图上画框设置裁剪范围")
        else:
            self.duration_label.setStyleSheet("color: #f44336; font-size: 9pt;")
            self.duration_label.setText("❌ 无法打开视频文件")
            QMessageBox.warning(self, "错误", f"无法打开视频文件:\n{path}")

    def _browse_output_file(self) -> None:
        """浏览选择输出路径。"""
        # 根据编码器设置确定输出文件扩展名
        codec_ext_map = {
            "mp4v": ".mp4",
            "XVID": ".avi",
            "avc1": ".mkv",
        }
        current_codec = self.codec_combo.currentData() if hasattr(self, 'codec_combo') else "mp4v"
        output_ext = codec_ext_map.get(current_codec, ".mp4")

        if self._video_path:
            default_path = Path(self._video_path)
            default_name = f"{default_path.stem}_stabilized{output_ext}"
            # 默认输出到watermarked目录
            default_dir = self._default_output_dir or str(default_path.parent)
        else:
            default_name = f"output_stabilized{output_ext}"
            default_dir = self._default_output_dir or self._default_dir

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存稳定后的视频",
            os.path.join(default_dir, default_name),
            "MP4 (*.mp4);;AVI (*.avi);;MKV (*.mkv);;所有文件 (*)"
        )
        if file_path:
            self.output_input.setText(file_path)

    def _on_preview_load_progress(self, current: int, total: int) -> None:
        """预览帧加载进度回调。"""
        pct = int(current / total * 100) if total > 0 else 0
        self.duration_label.setText(f"⏳ 加载预览帧... {pct}% ({current}/{total})")

    def _on_preview_loaded(self, frames: list, fps: float) -> None:
        """预览帧加载完成回调。"""
        if frames:
            self.preview_widget.set_frames(frames)
            self.preview_widget.set_fps(fps)
            self.duration_label.setStyleSheet("color: #4CAF50; font-size: 9pt; font-weight: bold;")

            # ★ 获取时间范围设置
            start_time = self.start_time_spin.value()
            end_time = self.end_time_spin.value()

            # 显示时间范围信息
            if end_time > 0:
                time_range_str = f"时间范围: {start_time:.1f}s - {end_time:.1f}s"
            else:
                time_range_str = f"起始时间: {start_time:.1f}s"

            # 恢复显示视频信息（从file_input获取路径重新读取）
            if self._video_path:
                cap = cv2.VideoCapture(self._video_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = total_frames / fps if fps > 0 else 0
                    self.duration_label.setText(
                        f"{time_range_str} | {duration:.1f}秒总长 ({total_frames}帧, {width}×{height}, {fps:.1f}fps) ✓ 预览就绪({len(frames)}帧)"
                    )
                    cap.release()
            print(f"[视频稳定GUI] 预览加载完成，共{len(frames)}帧 (时间范围: {time_range_str})", flush=True)

    def _on_preview_load_error(self, error_msg: str) -> None:
        """预览帧加载失败回调。"""
        self.duration_label.setStyleSheet("color: #f44336; font-size: 9pt;")
        self.duration_label.setText(f"⚠ 预览加载失败: {error_msg}")
        print(f"[视频稳定GUI] ⚠ 预览加载失败: {error_msg}", flush=True)

    def _start_preview_loader(self, video_path: str) -> None:
        """启动后台线程加载原始视频采样帧（用于播放预览）。

        每秒取1帧，最多30帧，轻量级不卡UI。
        """
        # 如果已有加载线程在运行，先停止
        if self._preview_loader and self._preview_loader.isRunning():
            self._preview_loader.quit()
            self._preview_loader.wait()

        start_time = self.start_time_spin.value() if hasattr(self, 'start_time_spin') else 0.0
        end_time = self.end_time_spin.value() if hasattr(self, 'end_time_spin') else -1.0

        self._preview_loader = VideoPreviewWorker(
            video_path,
            start_time=start_time,
            end_time=end_time,
            parent=self
        )
        self._preview_loader.progress.connect(self._on_preview_load_progress)
        self._preview_loader.finished.connect(self._on_preview_loaded)
        self._preview_loader.error.connect(self._on_preview_load_error)
        self._preview_loader.start()

    def _preview_first_frame(self) -> None:
        """预览处理效果（实际执行防抖+裁剪+图像调整，显示处理后结果）。"""
        if not self._video_path:
            QMessageBox.information(self, "提示", "请先选择视频文件")
            return

        # 收集当前参数
        options = self._collect_options()

        # 显示处理提示
        self.duration_label.setText("⏳ 正在生成预览（执行防抖+裁剪+调整）...")
        self.duration_label.setStyleSheet("color: #ff9800; font-size: 10pt; font-weight: bold;")

        # 启动后台线程执行实际处理（输出到内存用于预览）
        self._preview_processor = VideoPreviewProcessor(
            self._video_path,
            options,
            self
        )
        self._preview_processor.progress.connect(self._on_preview_process_progress)
        self._preview_processor.finished.connect(self._on_preview_processed)
        self._preview_processor.error.connect(self._on_preview_process_error)
        self._preview_processor.start()

        print(f"[视频稳定GUI] 开始生成预览效果...", flush=True)

    def _on_preview_process_progress(self, current: int, total: int, stage: str = "") -> None:
        """预览处理进度回调。"""
        pct = int(current / total * 100) if total > 0 else 0
        if stage:
            self.duration_label.setText(f"⏳ 预览处理 [{stage}]... {pct}% ({current}/{total})")
        else:
            self.duration_label.setText(f"⏳ 预览处理中... {pct}% ({current}/{total})")

    def _on_preview_processed(self, frames: list, fps: float, info: dict) -> None:
        """预览处理完成回调（显示处理后的帧）。"""
        if frames:
            self.preview_widget.set_frames(frames)
            self.preview_widget.set_fps(fps)
            self.duration_label.setStyleSheet("color: #4CAF50; font-size: 9pt; font-weight: bold;")

            # 显示处理信息
            start_time = info.get('start_time', 0)
            end_time = info.get('end_time', -1)
            if end_time > 0:
                time_range_str = f"{start_time:.1f}s - {end_time:.1f}s"
            else:
                time_range_str = f"从{start_time:.1f}s开始"

            applied_effects = []
            if info.get('stabilized'):
                applied_effects.append("防抖")
            if info.get('cropped'):
                applied_effects.append("裁剪")
            if info.get('adjusted'):
                applied_effects.append("调整")

            effects_str = ", ".join(applied_effects) if applied_effects else "原始"

            self.duration_label.setText(
                f"✓ 预览就绪({len(frames)}帧) | 时间: {time_range_str} | 效果: {effects_str}"
            )
            print(f"[视频稳定GUI] 预览处理完成！共{len(frames)}帧 (效果: {effects_str})", flush=True)
        else:
            self.duration_label.setText("⚠ 预览处理失败：无输出帧")
            self.duration_label.setStyleSheet("color: #f44336; font-size: 9pt;")

    def _on_preview_process_error(self, error_msg: str) -> None:
        """预览处理失败回调。"""
        self.duration_label.setStyleSheet("color: #f44336; font-size: 9pt;")
        self.duration_label.setText(f"⚠ 预览处理失败: {error_msg}")
        print(f"[视频稳定GUI] ⚠ 预览处理失败: {error_msg}", flush=True)

    def _on_time_range_changed(self) -> None:
        """时间范围改变时提示用户重新预览。"""
        if self._video_path:
            self.duration_label.setStyleSheet("color: #ff9800; font-size: 9pt;")
            self.duration_label.setText("参数已更改，点击[预览]查看效果")

    def _toggle_playback(self) -> None:
        """切换播放/暂停状态。"""
        if self.preview_widget._is_playing:
            self.preview_widget.pause()
            self.play_btn.setText("▶ 播放")
        else:
            if not self._video_path:
                QMessageBox.information(self, "提示", "请先选择视频文件")
                return
            ok = self.preview_widget.play()
            if ok:
                self.play_btn.setText("⏸ 暂停")
            else:
                QMessageBox.information(self, "提示", "无法播放视频，请检查视频文件是否正常")

    def _stop_playback(self) -> None:
        """停止播放并回到首帧。"""
        self.preview_widget.stop()
        self.play_btn.setText("▶ 播放")

    def _on_playback_speed_changed(self, index: int) -> None:
        """播放倍速改变回调。"""
        speed_map = [0.5, 1.0, 2.0, 4.0, 8.0]  # 对应选项: 慢/原速/2x/4x/8x
        speed = speed_map[index] if 0 <= index < len(speed_map) else 1.0
        self.preview_widget.set_playback_speed(speed)
        print(f"[视频稳定GUI] 播放速度: {speed}x (基于原视频速度)", flush=True)

    def _seek_frame(self, value: int) -> None:
        """跳转到指定帧（进度条拖动）。"""
        info = self.preview_widget.get_video_info()
        total = info["total_frames"]
        if total <= 1:
            return
        target = int(value * total / 100)
        target = max(0, min(target, total - 1))
        if info["mode"] == "video":
            self.preview_widget.seek_video(target)
        else:
            self.preview_widget._current_idx = target
            self.preview_widget._current_frame = self.preview_widget._frames[target]
            self.preview_widget._original_frame = self.preview_widget._frames[target]
            self.preview_widget.update()

    def _collect_options(self) -> VideoStabilizeOptions:
        """收集UI参数为选项对象。"""
        opts = VideoStabilizeOptions(
            algorithm=self.algo_combo.currentData(),
            stabilizer_mode=self.mode_combo.currentData() if hasattr(self, 'mode_combo') else "standard",
            start_time=self.start_time_spin.value(),
            end_time=self.end_time_spin.value(),
            crop_left=self.crop_left_spin.value(),
            crop_right=self.crop_right_spin.value(),
            crop_top=self.crop_top_spin.value(),
            crop_bottom=self.crop_bottom_spin.value(),
            # ===== vidstab 防抖参数 =====
            smoothing_window=self.smooth_window_spin.value(),
            border_type=self.border_type_combo.currentData(),
            border_size=-1,  # 自动根据 trim_ratio 计算
            trim_ratio=self.trim_ratio_spin.value(),
            feature_detector="GFTT",
            smoothing_radius=self.smooth_window_spin.value() // 2,  # 内置算法使用
            mtools_motion_profile="handheld",
            mtools_stability_strength=70.0,
            # ===== 音频设置 =====
            keep_audio=self.keep_audio_check.isChecked(),
            # ===== 图像调整参数 =====
            brightness=self.brightness_slider.value() / 10.0,
            contrast=self.contrast_slider.value() / 10.0,
            saturation=self.saturation_slider.value() / 10.0,
            sharpness=self.sharpness_slider.value() / 10.0,
            exposure=self.exposure_slider.value() / 10.0,
            highlights=self.highlights_slider.value() / 10.0,
            shadows=self.shadows_slider.value() / 10.0,
            temperature=self.temperature_slider.value() / 10.0,
            tint=self.tint_slider.value() / 10.0,
            output_fps=-1.0,
            output_codec=self.codec_combo.currentData(),
            output_quality=self.quality_spin.value(),
        )
        return opts

    def _start_processing(self) -> None:
        """开始视频稳定处理。"""
        if not self._video_path:
            QMessageBox.warning(self, "错误", "请先选择视频文件")
            return

        output_path = self.output_input.text().strip()
        if not output_path:
            QMessageBox.warning(self, "错误", "请指定输出路径")
            return

        # 收集参数
        self._options = self._collect_options()

        # 更新UI状态
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.log_text.clear()
        self.log_text.append("=" * 50)
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] 开始处理...")
        self.log_text.append(f"输入: {self._video_path}")
        self.log_text.append(f"输出: {output_path}")
        self.log_text.append(f"算法: {self._options.algorithm}")
        self.log_text.append("=" * 50)

        # 启动工作线程
        self._worker = VideoStabilizeWorker(
            self._video_path,
            output_path,
            self._options,
            self
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._on_log)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _cancel_processing(self) -> None:
        """取消处理。"""
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(3000)
            self.log_text.append("\n[已取消]")
            self._reset_ui_state()

    def _on_progress(self, current: int, total: int) -> None:
        """进度回调。"""
        pct = int(current / max(total, 1) * 100)
        self.log_text.append(f"处理进度: {pct}% ({current}/{total})")

    def _on_log(self, msg: str) -> None:
        """日志回调。"""
        self.log_text.append(msg)
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_finished(self, output_path: str) -> None:
        """处理完成回调。"""
        self.log_text.append("\n" + "=" * 50)
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] ✓ 处理完成！")
        self.log_text.append(f"输出文件: {output_path}")
        self.log_text.append("=" * 50)

        QMessageBox.information(
            self,
            "完成",
            f"视频稳定处理完成！\n\n输出文件:\n{output_path}"
        )

        self._reset_ui_state()

        # 尝试打开输出文件夹
        try:
            output_dir = os.path.dirname(output_path)
            if os.path.isdir(output_dir):
                os.startfile(output_dir)  # Windows
        except Exception:
            pass

    def _on_failed(self, error: str) -> None:
        """处理失败回调。"""
        self.log_text.append(f"\n✗ 错误: {error}")
        QMessageBox.critical(
            self,
            "处理失败",
            f"视频稳定处理失败:\n\n{error}"
        )
        self._reset_ui_state()

    def _reset_ui_state(self) -> None:
        """重置UI状态。"""
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._worker = None

    def _get_state_file(self) -> str:
        """获取状态文件路径。"""
        import os
        # 保存到用户目录下的 .video_stab_state.json
        state_dir = os.path.join(os.path.expanduser("~"), ".birdy")
        os.makedirs(state_dir, exist_ok=True)
        return os.path.join(state_dir, "video_stab_state.json")

    def _load_state(self) -> None:
        """从JSON文件加载上次保存的参数。"""
        import json
        import os

        state_file = self._get_state_file()
        if not os.path.exists(state_file):
            return

        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            print(f"[视频稳定GUI] 加载已保存的参数", flush=True)

            # 恢复时间范围
            if 'start_time' in state:
                self.start_time_spin.setValue(state['start_time'])
            if 'end_time' in state:
                self.end_time_spin.setValue(state['end_time'])

            # 恢复空间裁剪
            if 'crop_left' in state:
                self.crop_left_spin.setValue(state['crop_left'])
            if 'crop_right' in state:
                self.crop_right_spin.setValue(state['crop_right'])
            if 'crop_top' in state:
                self.crop_top_spin.setValue(state['crop_top'])
            if 'crop_bottom' in state:
                self.crop_bottom_spin.setValue(state['crop_bottom'])

            # 恢复图像调整参数
            if 'brightness' in state:
                self.brightness_slider.setValue(state['brightness'])
            if 'contrast' in state:
                self.contrast_slider.setValue(state['contrast'])
            if 'saturation' in state:
                self.saturation_slider.setValue(state['saturation'])
            if 'sharpness' in state:
                self.sharpness_slider.setValue(state['sharpness'])
            if 'exposure' in state:
                self.exposure_slider.setValue(state['exposure'])
            if 'highlights' in state:
                self.highlights_slider.setValue(state['highlights'])
            if 'shadows' in state:
                self.shadows_slider.setValue(state['shadows'])
            if 'temperature' in state:
                self.temperature_slider.setValue(state['temperature'])
            if 'tint' in state:
                self.tint_slider.setValue(state['tint'])

            # 恢复其他设置
            if 'smoothing_window' in state:
                self.smooth_window_spin.setValue(state['smoothing_window'])
            if 'border_type' in state and hasattr(self, 'border_type_combo'):
                idx = self.border_type_combo.findData(state['border_type'])
                if idx >= 0:
                    self.border_type_combo.setCurrentIndex(idx)
            if 'stabilizer_mode' in state and hasattr(self, 'mode_combo'):
                idx = self.mode_combo.findData(state['stabilizer_mode'])
                if idx >= 0:
                    self.mode_combo.setCurrentIndex(idx)
            if 'keep_audio' in state and hasattr(self, 'keep_audio_check'):
                self.keep_audio_check.setChecked(bool(state['keep_audio']))
            if 'codec_index' in state and hasattr(self, 'codec_combo'):
                idx = min(state['codec_index'], self.codec_combo.count() - 1)
                self.codec_combo.setCurrentIndex(idx)
            if 'speed_index' in state and hasattr(self, 'speed_combo'):
                idx = min(state['speed_index'], self.speed_combo.count() - 1)
                self.speed_combo.setCurrentIndex(idx)

            # 触发一次图像调整更新
            self._on_image_adjust_changed()

        except Exception as e:
            print(f"[视频稳定GUI] ⚠ 加载状态失败: {e}", flush=True)

    def _save_state(self) -> None:
        """将当前参数保存到JSON文件。"""
        import json

        try:
            state = {
                # 时间范围
                'start_time': self.start_time_spin.value(),
                'end_time': self.end_time_spin.value(),

                # 空间裁剪
                'crop_left': self.crop_left_spin.value(),
                'crop_right': self.crop_right_spin.value(),
                'crop_top': self.crop_top_spin.value(),
                'crop_bottom': self.crop_bottom_spin.value(),

                # 图像调整参数
                'brightness': self.brightness_slider.value(),
                'contrast': self.contrast_slider.value(),
                'saturation': self.saturation_slider.value(),
                'sharpness': self.sharpness_slider.value(),
                'exposure': self.exposure_slider.value(),
                'highlights': self.highlights_slider.value(),
                'shadows': self.shadows_slider.value(),
                'temperature': self.temperature_slider.value(),
                'tint': self.tint_slider.value(),

                # 其他设置
                'smoothing_window': self.smooth_window_spin.value(),
                'border_type': self.border_type_combo.currentData(),
                'stabilizer_mode': self.mode_combo.currentData() if hasattr(self, 'mode_combo') else 'standard',
                'keep_audio': self.keep_audio_check.isChecked(),
            }

            # 保存下拉框索引
            if hasattr(self, 'codec_combo'):
                state['codec_index'] = self.codec_combo.currentIndex()
            if hasattr(self, 'speed_combo'):
                state['speed_index'] = self.speed_combo.currentIndex()

            state_file = self._get_state_file()
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

            print(f"[视频稳定GUI] 参数已保存", flush=True)

        except Exception as e:
            print(f"[视频稳定GUI] ⚠ 保存状态失败: {e}", flush=True)

    def closeEvent(self, event):
        """窗口关闭时自动保存参数。"""
        self._save_state()
        super().closeEvent(event)


# 从外部调用的入口函数
def open_video_stabilize_dialog(parent=None, default_dir: str = "", default_output_dir: str = "") -> int:
    """
    打开视频稳定对话框。

    Args:
        parent: 父窗口
        default_dir: 默认文件浏览目录（用于选择输入视频）
        default_output_dir: 默认输出目录（watermarked目录）

    Returns:
        QDialog.Accepted 或 QDialog.Rejected
    """
    dialog = VideoStabilizeDialog(parent, default_dir=default_dir, default_output_dir=default_output_dir)
    result = dialog.exec_()
    return result


if __name__ == "__main__":
    # 测试用
    app = QApplication(sys.argv)
    dlg = VideoStabilizeDialog(default_dir=r"C:\Users\brigc\Pictures")
    dlg.show()
    sys.exit(app.exec_())
