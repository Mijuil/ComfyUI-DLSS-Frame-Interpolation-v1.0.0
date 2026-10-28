from __future__ import annotations

from contextlib import suppress
import json
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, InputImpl, Types, io, ui

from .dlss_engine.core.composition import (
    TemporalDeflicker,
    apply_color_grade,
    apply_enhance,
    color_grade_report,
    compose_sdr,
    composition_report,
    enhance_report,
    validate_color_grade,
    validate_detail_strength,
    validate_enhance,
)
from .dlss_engine.core.ffmpeg import CODEC_CHOICES, ENCODING_QUALITIES
from .dlss_engine.core.gpu_selection import resolve_runtime_ai_gpu
from .dlss_engine.core.jobs import active_job, cancel_active_job
from .dlss_engine.core.runtime import (
    DLSS_MODEL_PRESETS,
    NR_PRESETS,
    NR_STYLES,
    UPSCALING_MODES,
    DLSSFrameSession,
    prepare_runtime,
    resize_fit,
    resolve_native_settings,
    resolve_output_size,
    resolve_upscaling_mode,
    verify_feature_18,
)
from .dlss_engine.frame_interpolation import ENGINE_CHOICES, FPS_CHOICES, FrameInterpolationOptions, interpolate_video
from .dlss_engine.video import ConversionOptions, convert_video


CONTAINER_CHOICES = ("MP4", "MKV", "MOV")
RENAME_MODES = ("Auto", "Copy", "Custom")

# Canvas-only cosmetics (section header banners) served to the frontend.
WEB_DIRECTORY = "web"
UPSCALE_FACTORS = {mode["label"]: factor for factor, mode in UPSCALING_MODES.items()}


def _progress_callback():
    progress_bar = comfy.utils.ProgressBar(1000)

    def progress(value: float, _message: str) -> None:
        try:
            comfy.model_management.throw_exception_if_processing_interrupted()
        except comfy.model_management.InterruptProcessingException:
            cancel_active_job()
            raise
        progress_bar.update_absolute(round(float(value) * 1000), 1000)

    return progress_bar, progress


def _temporary_video_input(video: io.Video.Type, prefix: str):
    source = video.get_stream_source()
    start_time, duration = video.get_active_trim_window()
    if isinstance(source, str) and start_time == 0.0 and duration == 0.0:
        return suppress(), Path(source).resolve()
    temp = tempfile.TemporaryDirectory(prefix=prefix, dir=folder_paths.get_temp_directory())
    input_path = Path(temp.name) / "input.mkv"
    video.save_to(str(input_path), format=Types.VideoContainer.MKV, codec=Types.VideoCodec.AUTO)
    return temp, input_path


def _preview_output(output_path: Path, report_text: str) -> io.NodeOutput:
    temp_base = Path(folder_paths.get_temp_directory()).resolve()
    relative = output_path.relative_to(temp_base)
    preview = ui.SavedResult(output_path.name, relative.parent.as_posix(), io.FolderType.temp)
    return io.NodeOutput(
        InputImpl.VideoFromFile(str(output_path)),
        report_text,
        ui=ui.PreviewVideo([preview]),
    )


def _neural_options(
    nr_preset: str,
    nr_style: str,
    nr_intensity: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    automatic_mask: bool,
    dlss_model_preset: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        nr_preset=str(nr_preset),
        nr_style=str(nr_style),
        nr_intensity=float(nr_intensity),
        local_tone_strength=float(local_tone_strength),
        local_structure_strength=float(local_structure_strength),
        skin_structure_strength=float(skin_structure_strength),
        automatic_mask=bool(automatic_mask),
        dlss_model_preset=str(dlss_model_preset),
    )


def _neural_inputs() -> list:
    """DLSS5 native neural controls; the section toggle gates them all."""
    return [
        io.Boolean.Input(
            "enable_nr", default=True,
            display_name="Нейрообработка DLSS5",
            tooltip="ВЫКЛЮЧАТЕЛЬ РАЗДЕЛА «Нейрообработка DLSS5». "
            "Выключите — все настройки ниже игнорируются (шумодав работает в заводском режиме).",
        ),
        io.Combo.Input(
            "nr_preset", options=list(NR_PRESETS), default="Default",
            tooltip="Пресет нейрошумодава из DLSS5 SDK: Default — рекомендованный NVIDIA; "
            "остальные — альтернативные заводские конфигурации.",
        ),
        io.Combo.Input(
            "nr_style", options=list(NR_STYLES), default="Default",
            tooltip="Стиль обработки шумодава: Default — стандартный баланс; "
            "Natural — мягче, бережнее к текстуре кожи; "
            "Cinematic — плотнее глушит шум, кинематографичный результат.",
        ),
        io.Float.Input(
            "nr_intensity", default=1.0, min=0.0, max=2.0, step=0.05,
            tooltip="Сила шумоподавления: 0 — без шумодава; 1 — штатный режим NVIDIA; "
            "до 2 — агрессивнее (риск потери мелких деталей и текстуры кожи).",
        ),
        io.Float.Input(
            "local_tone_strength", default=1.0, min=0.0, max=2.0, step=0.05,
            tooltip="Локальный тон: как активно DLSS выравнивает яркость соседних областей. "
            "Меньше — естественнее градиенты неба и теней.",
        ),
        io.Float.Input(
            "local_structure_strength", default=1.0, min=0.0, max=2.0, step=0.05,
            tooltip="Сохранение структур: контуры, текст, узоры. "
            "Выше — чётче мелкие детали; ниже — мягче, меньше «звона» на границах.",
        ),
        io.Float.Input(
            "skin_structure_strength", default=-1.0, min=-1.0, max=2.0, step=0.05,
            tooltip="Текстура кожи: −1 — авто (DLSS сам распознаёт кожу и бережёт её); "
            "0 — выключено; >0 — ручная сила сохранения пор и микрорельефа лица.",
        ),
        io.Boolean.Input(
            "automatic_mask", default=False,
            tooltip="Автомаска: DLSS сам находит зоны (кожа, волосы, глубокие тени), "
            "где обработка становится деликатнее.",
        ),
        io.Combo.Input(
            "dlss_model_preset", options=list(DLSS_MODEL_PRESETS), default="Default",
            tooltip="Версия нейросети DLSS (модели J/K/L/M). Баланс детализации и "
            "стабильности различается; Default — модель вашего драйвера.",
        ),
    ]


def _enhance_inputs() -> list:
    """Post-DLSS cleanup & detail controls, every feature has its own toggle."""
    return [
        io.Boolean.Input(
            "enable_enhance", default=False,
            display_name="Качество и очистка",
            tooltip="ВЫКЛЮЧАТЕЛЬ РАЗДЕЛА «Качество и очистка»: гамма, кожа, дебандинг, "
            "дефликер, чёткость. Выкл — блок полностью обойдён, картинка как из DLSS.",
        ),
        io.Boolean.Input(
            "enable_deband", default=False,
            tooltip="Дебандинг: убирает полосы и блочность H.264/H.265 на градиентах "
            "(небо, тени). Включайте для пережатых или сгенерированных видео.",
        ),
        io.Float.Input(
            "gamma", default=1.0, min=0.25, max=2.5, step=0.05, optional=True,
            tooltip="Гамма средних тонов: <1 — осветляет кожу и полутени (лечит «тёмную кожу» "
            "после DLSS); >1 — затемняет. 1.0 — без изменений.",
        ),
        io.Float.Input(
            "skin_smoothing", default=0.0, min=0.0, max=1.0, step=0.05, optional=True,
            tooltip="Сглаживание кожи: двусторонний фильтр убирает генерационный шум и рябь "
            "на коже, сохраняя поры и контуры. 0 — выкл; 0.3–0.5 — лёгкая чистка.",
        ),
        io.Float.Input(
            "clarity", default=0.0, min=-1.0, max=1.0, step=0.05, optional=True,
            tooltip="Клэрити (локальный контраст средних частот): + — возвращает объём и "
            "резкость после шумодава; − — мягче/пластиковее. 0 — выкл.",
        ),
        io.Float.Input(
            "sharpen", default=0.0, min=0.0, max=2.0, step=0.05, optional=True,
            tooltip="Резкость (unsharp mask). Начните с 0.3–0.6; выше 1.0 — риск ореолов.",
        ),
        io.Boolean.Input(
            "edge_only", default=True,
            tooltip="Только контуры: резкость действует лишь на границы (Sobel-маска); "
            "плоские зоны — кожа, небо — не затрагиваются. Рекомендуется держать вкл.",
        ),
        io.Float.Input(
            "temporal_deflicker", default=0.0, min=0.0, max=1.0, step=0.05, optional=True,
            tooltip="Анти-мерцание (только видео): стабилизирует скачки яркости между кадрами "
            "после DLSS. 0 — выкл; 0.5–0.8 — мягкая стабилизация. На статичном свете почти не влияет.",
        ),
    ]


def _output_detail_strength_input():
    """Append-only widget so saved video workflows retain their existing order."""
    return io.Float.Input(
        "output_detail_strength", default=1.0, min=1.0, max=2.0,
        step=0.05, optional=True, display_name="Композиция вывода",
        tooltip="SDR-композиция вывода: 1 — сохраняет выход DLSS как есть; "
        "2 — усиливает изменения яркости. Отдельно от силы шумодава.",
    )


def _color_grade_inputs():
    """Append-only color & light widgets; all neutral at 0.0 = no grading."""
    return [
        io.Boolean.Input(
            "enable_color", default=False,
            display_name="Цвет и свет",
            tooltip="ВЫКЛЮЧАТЕЛЬ РАЗДЕЛА «Цвет и свет»: luma-выравнивание и вся цветокоррекция. "
            "Выкл — цвет не трогается вовсе.",
        ),
        io.Float.Input("luma_match", default=0.0, min=0.0, max=1.0, step=0.05,
                       optional=True,
                       tooltip="Авто-исправление потемнения: подтягивает яркость результата к "
                       "источнику (0 — выкл, 1 — полная коррекция). Для кожи рекомендую 0.5–0.8."),
        io.Float.Input("exposure", default=0.0, min=-1.0, max=1.0, step=0.05,
                       optional=True, tooltip="Ручная экспозиция в стопах (−1..+1): "
                       "проще говоря, общая яркость кадра."),
        io.Float.Input("contrast", default=0.0, min=-1.0, max=1.0, step=0.05,
                       optional=True, tooltip="Контраст вокруг среднего серого: "
                       "+ — сочнее и жёстче, − — мягче, туманнее."),
        io.Float.Input("saturation", default=0.0, min=-1.0, max=1.0, step=0.05,
                       optional=True, tooltip="Насыщенность цвета: + — ярче цвета, "
                       "− — ближе к ч/б."),
        io.Float.Input("shadows", default=0.0, min=-1.0, max=1.0, step=0.05,
                       optional=True, tooltip="Тени: + — поднять (появится детали в тёмном), "
                       "− — придавить (глубже чёрный)."),
        io.Float.Input("highlights", default=0.0, min=-1.0, max=1.0, step=0.05,
                       optional=True, tooltip="Света: + — вернуть детали в пересветах, "
                       "− — сжечь ярче."),
    ]


class NvidiaDLSSFrameInterpolation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NvidiaDLSSFrameInterpolation",
            display_name="NVIDIA DLSS Frame Interpolation",
            category="video/interpolation",
            description="Interpolates a VIDEO with the bundled NVIDIA DLSS Frame Generation runtime. Connect the VIDEO output to ComfyUI's Save Video node.",
            inputs=[
                io.Video.Input("video", tooltip="Video to interpolate."),
                io.Combo.Input("output_fps", options=list(FPS_CHOICES), default="60", tooltip="Fractional choices use exact 1001-based rates."),
                io.Combo.Input("dlss_engine", options=list(ENGINE_CHOICES), default="Auto", tooltip="Auto uses an exact native grid when supported, then cascades when required."),
                io.Combo.Input("encoding_quality", options=list(ENCODING_QUALITIES), default="Max"),
                io.Combo.Input("video_codec", options=list(CODEC_CHOICES), default="H.264", tooltip="Plain codecs use CPU encoding; NVIDIA NVENC choices use the GPU."),
                io.Combo.Input("container", options=list(CONTAINER_CHOICES), default="MP4"),
                io.Combo.Input("rename", options=list(RENAME_MODES), default="Auto", tooltip="Controls the temporary VIDEO filename only. ComfyUI's Save Video node owns the final name."),
                io.String.Input("custom_suffix", default="_DLSSFG", tooltip="Used for the temporary filename only when Rename is Custom."),
                io.Boolean.Input("hdr_mode", default=False, tooltip="10-bit output with input colorspace metadata. Supported by H.265, AV1, and ProRes only."),
            ],
            outputs=[
                io.Video.Output("video", display_name="interpolated_video"),
                io.String.Output("report", display_name="report_json"),
            ],
        )

    @classmethod
    def cancel_current_job(cls) -> str:
        return cancel_active_job()

    @classmethod
    def execute(
        cls,
        video: io.Video.Type,
        output_fps: str,
        dlss_engine: str,
        encoding_quality: str,
        video_codec: str,
        container: str,
        rename: str,
        custom_suffix: str,
        hdr_mode: bool,
    ) -> io.NodeOutput:
        _progress_bar, progress = _progress_callback()
        temp_base = Path(folder_paths.get_temp_directory()).resolve()
        output_dir = Path(tempfile.mkdtemp(prefix="dlssfg-output-", dir=temp_base))
        options = FrameInterpolationOptions(
            target_fps=str(output_fps),
            engine=str(dlss_engine),
            codec=str(video_codec),
            container=str(container),
            quality=str(encoding_quality),
            hdr_mode=bool(hdr_mode),
            rename_mode=str(rename),
            custom_suffix=str(custom_suffix),
        )
        try:
            input_context, input_path = _temporary_video_input(video, "dlssfg-input-")
            with input_context:
                result = interpolate_video(
                    input_path,
                    options,
                    progress,
                    output_directory=output_dir,
                    jobs_directory=temp_base / "dlss_frame_interpolation_jobs",
                    logs_directory=output_dir / "reports",
                )
            output_path = Path(result.output_path).resolve()
            report_text = Path(result.report_path).read_text(encoding="utf-8")
            json.loads(report_text)
            return _preview_output(output_path, report_text)
        except BaseException:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise


class NvidiaDLSSVideoUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NvidiaDLSSVideoUpscale",
            display_name="NVIDIA DLSS Video Upscale",
            category="video/upscaling",
            description="Upscales a VIDEO through the bundled DLSS 5 feature-18 pipeline. The temporary VIDEO output connects to ComfyUI's Save Video node.",
            inputs=[
                io.Video.Input("video"),
                io.Combo.Input("upscale_mode", options=list(UPSCALE_FACTORS), default="1.5× (Quality)"),
                io.Boolean.Input("require_neural_upscaling", default=False, tooltip="Fail instead of returning a larger fallback result when NVIDIA reports neural upscaling inactive."),
                *_neural_inputs(),
                io.Combo.Input("encoding_quality", options=list(ENCODING_QUALITIES), default="Max"),
                io.Combo.Input("video_codec", options=list(CODEC_CHOICES), default="H.264"),
                io.Combo.Input("container", options=list(CONTAINER_CHOICES), default="MP4"),
                io.Combo.Input("rename", options=list(RENAME_MODES), default="Auto", tooltip="Controls only the temporary filename."),
                io.String.Input("custom_suffix", default="_DLSS5"),
                io.Boolean.Input("hdr_mode", default=False, tooltip="10-bit output with input colorspace metadata. Supported by H.265, AV1, and ProRes only."),
                _output_detail_strength_input(),
                *_enhance_inputs(),
                *_color_grade_inputs(),
            ],
            outputs=[
                io.Video.Output("video", display_name="upscaled_video"),
                io.String.Output("report", display_name="report_json"),
            ],
        )

    @classmethod
    def cancel_current_job(cls) -> str:
        return cancel_active_job()

    @classmethod
    def execute(
        cls,
        video: io.Video.Type,
        upscale_mode: str,
        require_neural_upscaling: bool,
        nr_preset: str,
        nr_style: str,
        nr_intensity: float,
        local_tone_strength: float,
        local_structure_strength: float,
        skin_structure_strength: float,
        automatic_mask: bool,
        dlss_model_preset: str,
        encoding_quality: str,
        video_codec: str,
        container: str,
        rename: str,
        custom_suffix: str,
        hdr_mode: bool,
        output_detail_strength: float = 1.0,
        enable_nr: bool = True,
        enable_color: bool = False,
        enable_enhance: bool = False,
        luma_match: float = 0.0,
        exposure: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        shadows: float = 0.0,
        highlights: float = 0.0,
        enable_deband: bool = False,
        gamma: float = 1.0,
        skin_smoothing: float = 0.0,
        clarity: float = 0.0,
        sharpen: float = 0.0,
        edge_only: bool = True,
        temporal_deflicker: float = 0.0,
    ) -> io.NodeOutput:
        output_detail_strength = validate_detail_strength(output_detail_strength)
        if not enable_nr:
            nr_preset, nr_style = "Default", "Default"
            nr_intensity, local_tone_strength = 1.0, 1.0
            local_structure_strength, skin_structure_strength = 1.0, -1.0
            automatic_mask = False
            dlss_model_preset = "Default"
        if not enable_color:
            luma_match = exposure = contrast = saturation = shadows = highlights = 0.0
        color_grade = validate_color_grade(
            luma_match, exposure, contrast, saturation, shadows, highlights
        )
        if not enable_enhance:
            gamma, skin_smoothing, clarity, sharpen = 1.0, 0.0, 0.0, 0.0
            enable_deband, edge_only = False, True
            temporal_deflicker = 0.0
        enhance = validate_enhance(gamma, sharpen, clarity, skin_smoothing)
        enhance.edge_only = bool(edge_only)
        enhance.deband = bool(enable_deband)
        _progress_bar, progress = _progress_callback()
        temp_base = Path(folder_paths.get_temp_directory()).resolve()
        output_dir = Path(tempfile.mkdtemp(prefix="dlss5-video-output-", dir=temp_base))
        options = ConversionOptions(
            output_detail_strength=output_detail_strength,
            luma_match=color_grade.luma_match,
            exposure=color_grade.exposure,
            contrast=color_grade.contrast,
            saturation=color_grade.saturation,
            shadows=color_grade.shadows,
            highlights=color_grade.highlights,
            gamma=enhance.gamma,
            sharpen=enhance.sharpen,
            edge_only=enhance.edge_only,
            clarity=enhance.clarity,
            skin_smoothing=enhance.skin_smoothing,
            deband=enhance.deband,
            temporal_deflicker=temporal_deflicker,
            nr_preset=str(nr_preset),
            nr_style=str(nr_style),
            nr_intensity=float(nr_intensity),
            local_tone_strength=float(local_tone_strength),
            local_structure_strength=float(local_structure_strength),
            skin_structure_strength=float(skin_structure_strength),
            automatic_mask=bool(automatic_mask),
            dlss_model_preset=str(dlss_model_preset),
            upscaling_factor=UPSCALE_FACTORS[str(upscale_mode)],
            codec=str(video_codec),
            container=str(container),
            quality=str(encoding_quality),
            preserve_hdr=bool(hdr_mode),
            rename_mode=str(rename),
            custom_suffix=str(custom_suffix),
        )
        try:
            input_context, input_path = _temporary_video_input(video, "dlss5-video-input-")
            with input_context:
                result = convert_video(
                    input_path,
                    options,
                    progress,
                    output_directory=output_dir,
                    jobs_directory=temp_base / "dlss_video_upscale_jobs",
                    logs_directory=output_dir / "reports",
                )
            output_path = Path(result.output_path).resolve()
            report_path = Path(result.report_path).resolve()
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["node_policy"] = {"require_neural_upscaling": bool(require_neural_upscaling)}
            if require_neural_upscaling and options.upscaling_factor > 1.0 and not report["nr_upscaling_active"]:
                output_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "NVIDIA completed the frame processing but reported neural upscaling inactive. "
                    "Disable Require Neural Upscaling to accept the native fallback, or change the "
                    "source resolution, upscale mode, NVIDIA driver, or runtime configuration."
                )
            report_text = json.dumps(report, indent=2)
            return _preview_output(output_path, report_text)
        except BaseException:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise


class NvidiaDLSSImageUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NvidiaDLSSImageUpscale",
            display_name="NVIDIA DLSS Image Upscale",
            category="image/upscaling",
            description="Upscales a ComfyUI IMAGE batch directly in memory with the bundled DLSS 5 feature-18 pipeline.",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("upscale_mode", options=list(UPSCALE_FACTORS), default="1.5× (Quality)"),
                io.Boolean.Input("require_neural_upscaling", default=False, tooltip="Fail instead of returning a larger fallback result when NVIDIA reports neural upscaling inactive."),
                *_neural_inputs(),
                _output_detail_strength_input(),
                *_enhance_inputs(),
                *_color_grade_inputs(),
            ],
            outputs=[
                io.Image.Output("image", display_name="upscaled_image"),
                io.String.Output("report", display_name="report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        upscale_mode: str,
        require_neural_upscaling: bool,
        nr_preset: str,
        nr_style: str,
        nr_intensity: float,
        local_tone_strength: float,
        local_structure_strength: float,
        skin_structure_strength: float,
        automatic_mask: bool,
        dlss_model_preset: str,
        output_detail_strength: float = 1.0,
        enable_nr: bool = True,
        enable_color: bool = False,
        enable_enhance: bool = False,
        luma_match: float = 0.0,
        exposure: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        shadows: float = 0.0,
        highlights: float = 0.0,
        enable_deband: bool = False,
        gamma: float = 1.0,
        skin_smoothing: float = 0.0,
        clarity: float = 0.0,
        sharpen: float = 0.0,
        edge_only: bool = True,
        temporal_deflicker: float = 0.0,
    ) -> io.NodeOutput:
        output_detail_strength = validate_detail_strength(output_detail_strength)
        if not enable_nr:
            nr_preset, nr_style = "Default", "Default"
            nr_intensity, local_tone_strength = 1.0, 1.0
            local_structure_strength, skin_structure_strength = 1.0, -1.0
            automatic_mask = False
            dlss_model_preset = "Default"
        if not enable_color:
            luma_match = exposure = contrast = saturation = shadows = highlights = 0.0
        color_grade = validate_color_grade(
            luma_match, exposure, contrast, saturation, shadows, highlights
        )
        if not enable_enhance:
            gamma, skin_smoothing, clarity, sharpen = 1.0, 0.0, 0.0, 0.0
            enable_deband, edge_only = False, True
        enhance = validate_enhance(gamma, sharpen, clarity, skin_smoothing)
        enhance.edge_only = bool(edge_only)
        enhance.deband = bool(enable_deband)
        if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
            raise ValueError("IMAGE must have shape [batch, height, width, channels] with 1, 3, or 4 channels.")
        batch, input_height, input_width, channels = map(int, image.shape)
        if batch < 1 or input_width < 64 or input_height < 64:
            raise ValueError("DLSS image upscaling requires at least one image with width and height of 64 pixels or more.")

        factor, mode = resolve_upscaling_mode(UPSCALE_FACTORS[str(upscale_mode)])
        output_width, output_height = resolve_output_size(input_width, input_height, factor)
        native = resolve_native_settings(
            _neural_options(
                nr_preset,
                nr_style,
                nr_intensity,
                local_tone_strength,
                local_structure_strength,
                skin_structure_strength,
                automatic_mask,
                dlss_model_preset,
            )
        )
        prepared = prepare_runtime()
        gpu = resolve_runtime_ai_gpu(prepared.gpus, prepared.runtime_bundle, "auto")
        progress_bar = comfy.utils.ProgressBar(batch)
        session: DLSSFrameSession | None = None
        started = time.perf_counter()
        outputs: list[torch.Tensor] = []

        with active_job() as controller:
            try:
                session = DLSSFrameSession(
                    input_width=input_width,
                    input_height=input_height,
                    output_width=output_width,
                    output_height=output_height,
                    frame_count=batch,
                    warmup_frames=0,
                    factor=factor,
                    mode=mode,
                    native_settings=native,
                    gpu=gpu,
                    runtime_bundle=prepared.runtime_bundle,
                    controller=controller,
                )
                motion = np.zeros((session.render_height, session.render_width, 2), dtype=np.float16)
                source_batch = image.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).numpy()
                for index, source in enumerate(source_batch):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    pixels = np.rint(source * 255.0).astype(np.uint8)
                    if channels == 1:
                        rgb = np.repeat(pixels, 3, axis=2)
                        alpha = np.full((input_height, input_width), 255, dtype=np.uint8)
                    elif channels == 3:
                        rgb = pixels
                        alpha = np.full((input_height, input_width), 255, dtype=np.uint8)
                    else:
                        rgb = pixels[..., :3]
                        alpha = pixels[..., 3]
                    rgba = np.dstack((rgb, alpha))
                    prepared_rgba = resize_fit(rgba, session.render_width, session.render_height)
                    processed, _pts = session.process(
                        index=index,
                        rgba=prepared_rgba,
                        motion=motion,
                        reset=True,
                        pts=index,
                    )
                    processed = compose_sdr(
                        rgba, processed, output_detail_strength
                    )
                    processed = apply_color_grade(rgba, processed, color_grade)
                    processed = apply_enhance(processed, enhance)
                    if channels == 4:
                        processed[..., 3] = cv2.resize(
                            alpha,
                            (output_width, output_height),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                        result = processed
                    else:
                        result = processed[..., :3]
                    outputs.append(torch.from_numpy(np.ascontiguousarray(result)).to(dtype=torch.float32).div(255.0))
                    progress_bar.update_absolute(index + 1, batch)
                session.close()
                evidence = verify_feature_18(session.worker_logs, session.reshade_log_text())
            except Exception:
                if session is not None and not session.closed:
                    with suppress(Exception):
                        session.abort()
                raise

        elapsed = time.perf_counter() - started
        if require_neural_upscaling and factor > 1.0 and not evidence["nr_upscaling_active"]:
            raise RuntimeError(
                "NVIDIA completed the image processing but reported neural upscaling inactive. "
                "Disable Require Neural Upscaling to accept the native fallback, or change the "
                "source resolution, upscale mode, NVIDIA driver, or runtime configuration."
            )
        report = {
            "status": "success",
            "pipeline": "renodx-dlssnr-feature18",
            "feature_id": 18,
            "feature_18_confirmed": True,
            "images_processed": batch,
            "output_composition": composition_report(output_detail_strength),
            "color_grade": color_grade_report(color_grade),
            "enhance": enhance_report(enhance),
            "input_dimensions": {"width": input_width, "height": input_height},
            "negotiated_render_dimensions": {"width": session.render_width, "height": session.render_height},
            "output_dimensions": {"width": output_width, "height": output_height},
            "requested_upscaling_factor": factor,
            "dlss_mode": mode["name"],
            "requested_dlss_model_preset": str(dlss_model_preset),
            "applied_dlss_model_preset": session.applied_dlss_model_preset,
            "nr_upscaling_requested": factor > 1.0,
            "nr_upscaling_active": bool(evidence["nr_upscaling_active"]),
            "nr_native_fallback": bool(evidence["nr_native_fallback"]),
            "node_policy": {"require_neural_upscaling": bool(require_neural_upscaling)},
            "carrier_create_result": str(evidence["carrier_create_result"]),
            "native_settings": native,
            "gpu": gpu,
            "elapsed_seconds": elapsed,
            "dlssnr_evidence": evidence["evidence"],
            "worker_log": session.worker_logs,
            "worker_log_dropped_lines": session.worker_log_dropped_lines,
        }
        return io.NodeOutput(torch.stack(outputs), json.dumps(report, indent=2))


class DLSSVisualEnhancerExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [NvidiaDLSSFrameInterpolation, NvidiaDLSSVideoUpscale, NvidiaDLSSImageUpscale]


async def comfy_entrypoint() -> DLSSVisualEnhancerExtension:
    return DLSSVisualEnhancerExtension()


__all__ = [
    "NvidiaDLSSFrameInterpolation",
    "NvidiaDLSSVideoUpscale",
    "NvidiaDLSSImageUpscale",
    "comfy_entrypoint",
]
