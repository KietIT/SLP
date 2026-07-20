"""Interactive Streamlit demo for the three locked paper-v2 LoRA roles.

Run from the repository root with::

    python -m streamlit run scripts/demo_app.py

The app keeps all uploaded/recorded audio and result exports in memory. It does
not write into the formal paper-v2 artifact tree.
"""

from __future__ import annotations

import dataclasses
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs" / "demo.yaml"
METRICS = ("wer", "cer", "ter", "der", "fcer", "swdr")
ERROR_COUNTS = (
    "substitution",
    "deletion",
    "insertion",
    "tone",
    "diacritic",
    "final_consonant",
    "short_word_deletion",
)
ROLE_ORDER = ("ordinary_baseline", "selected_method", "locked_control")
ROLE_LABELS = {
    "ordinary_baseline": "Ordinary LoRA",
    "selected_method": "Tone-aware LoRA (lambda=0.05)",
    "locked_control": "Tone-aware LoRA (lambda=0.1)",
}
NOISE_LABELS = {
    "none": "Không trộn noise",
    "fan": "Quạt (synthetic)",
    "traffic": "Giao thông (synthetic)",
    "cafe": "Quán cà phê (synthetic)",
}


def _read_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid demo config: {CONFIG_PATH}")
    return data


CONFIG = _read_config()
APP_CONFIG = CONFIG["app"]
SAMPLE_RATE = int(APP_CONFIG["sample_rate"])
MAX_SECONDS = float(APP_CONFIG["max_audio_seconds"])


def _audio_bytes(upload: Any) -> bytes:
    if upload is None:
        raise ValueError("Chưa có audio đầu vào.")
    if hasattr(upload, "getvalue"):
        return bytes(upload.getvalue())
    if isinstance(upload, bytes):
        return upload
    raise TypeError(f"Unsupported audio input type: {type(upload).__name__}")


def _resample_linear(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Small dependency-free resampler suitable for a spoken demo clip."""

    if source_sr == target_sr or waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    output_length = max(1, int(round(waveform.size * target_sr / source_sr)))
    source_axis = np.linspace(0.0, 1.0, num=waveform.size, endpoint=False)
    target_axis = np.linspace(0.0, 1.0, num=output_length, endpoint=False)
    return np.interp(target_axis, source_axis, waveform).astype(np.float32)


def decode_audio(raw: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode, downmix, resample and enforce the formal 15-second limit."""

    with sf.SoundFile(io.BytesIO(raw)) as audio_file:
        source_sr = int(audio_file.samplerate)
        channels = int(audio_file.channels)
        waveform = audio_file.read(dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    waveform = _resample_linear(waveform, source_sr, SAMPLE_RATE)
    original_seconds = waveform.size / SAMPLE_RATE
    max_samples = int(round(MAX_SECONDS * SAMPLE_RATE))
    clipped = waveform.size > max_samples
    waveform = waveform[:max_samples]
    if waveform.size == 0:
        raise ValueError("Audio rỗng.")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("Audio chứa giá trị không hợp lệ (NaN/Inf).")
    return waveform.astype(np.float32), {
        "source_sample_rate": source_sr,
        "source_channels": channels,
        "original_seconds": round(original_seconds, 3),
        "processed_seconds": round(waveform.size / SAMPLE_RATE, 3),
        "clipped_to_max_seconds": clipped,
    }


def wav_bytes(waveform: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, min(int(width), values.size))
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def synthetic_noise(kind: str, length: int, seed: int = 42) -> np.ndarray:
    """Generate deterministic non-corpus noise for a qualitative live demo."""

    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32) / float(SAMPLE_RATE)
    white = rng.normal(0.0, 1.0, length).astype(np.float32)

    if kind == "fan":
        noise = 0.60 * np.sin(2 * np.pi * 120 * t)
        noise += 0.25 * np.sin(2 * np.pi * 240 * t)
        noise += 0.30 * _moving_average(white, 400)
    elif kind == "traffic":
        noise = _moving_average(white, 1200)
        noise += 0.40 * np.sin(2 * np.pi * 55 * t)
        if length:
            for center in rng.integers(0, length, size=max(3, int(length / SAMPLE_RATE * 2))):
                width = int(rng.integers(max(1, SAMPLE_RATE // 10), SAMPLE_RATE // 2))
                lo, hi = max(0, center - width), min(length, center + width)
                noise[lo:hi] += np.hanning(hi - lo).astype(np.float32) * rng.uniform(0.4, 1.0)
    elif kind == "cafe":
        noise = 0.45 * white
        noise += 0.25 * _moving_average(
            rng.normal(0.0, 1.0, length).astype(np.float32), 300
        )
        if length:
            for center in rng.integers(0, length, size=max(8, int(length / SAMPLE_RATE * 8))):
                width = int(rng.integers(80, 700))
                lo, hi = max(0, center - width), min(length, center + width)
                noise[lo:hi] += np.hanning(hi - lo).astype(np.float32) * rng.uniform(-0.8, 0.8)
    else:
        raise ValueError(f"Unknown synthetic noise type: {kind}")

    peak = float(np.max(np.abs(noise))) if noise.size else 0.0
    return (noise / peak if peak > 0 else noise).astype(np.float32)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    eps = 1e-8
    clean_power = float(np.mean(clean.astype(np.float64) ** 2)) + eps
    noise_power = float(np.mean(noise.astype(np.float64) ** 2)) + eps
    scale = np.sqrt(clean_power / (10.0 ** (snr_db / 10.0) * noise_power))
    mixed = clean + noise * np.float32(scale)
    peak = max(float(np.max(np.abs(mixed))), 1.0)
    return (mixed / peak).astype(np.float32)


def prepare_preview(clean: np.ndarray, noise_type: str, snr: int | None) -> np.ndarray:
    if noise_type == "none":
        return clean.copy()
    if snr is None:
        raise ValueError("SNR is required when noise is enabled.")
    noise = synthetic_noise(noise_type, clean.size, seed=int(APP_CONFIG["seed"]))
    return mix_at_snr(clean, noise, float(snr))


@st.cache_resource(show_spinner=False)
def load_engine() -> Any:
    from src.vitonesr.demo_inference import load_demo_engine

    return load_demo_engine(CONFIG_PATH)


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return dict(vars(value))
    return value


def normalise_results(payload: Any) -> list[dict[str, Any]]:
    # DemoResult also contains a waveform; avoid an unnecessary deep copy when
    # extracting the tabular rows from a dataclass result.
    if hasattr(payload, "rows"):
        payload = payload.rows
    else:
        payload = _plain(payload)
    if isinstance(payload, Mapping):
        for key in ("rows", "results", "predictions", "models"):
            if key in payload:
                payload = payload[key]
                break
        else:
            if all(isinstance(value, (Mapping, object)) for value in payload.values()):
                payload = [dict(_plain(value), role=key) for key, value in payload.items()]
            else:
                payload = [payload]
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        payload = [payload]

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        item = _plain(item)
        if not isinstance(item, Mapping):
            raise TypeError(f"Backend result #{index + 1} is not a mapping.")
        row = dict(item)
        nested_metrics = _plain(row.pop("metrics", {})) or {}
        nested_errors = _plain(row.pop("errors", row.pop("error_counts", {}))) or {}
        for name in METRICS:
            # Canonical backend metrics are fractions. Flat backend fields are
            # percentages for DataFrame convenience, so prefer the nested form.
            if isinstance(nested_metrics, Mapping) and name in nested_metrics:
                row[name] = nested_metrics.get(name)
        for name in ERROR_COUNTS:
            if name not in row and isinstance(nested_errors, Mapping):
                row[name] = nested_errors.get(name)
        role = str(row.get("role", row.get("role_id", row.get("id", ROLE_ORDER[index] if index < 3 else index))))
        row["role"] = role
        row["model"] = row.get("model", row.get("label", ROLE_LABELS.get(role, role)))
        row["transcript"] = row.get("transcript", row.get("hyp", row.get("text", "")))
        row["latency_seconds"] = row.get(
            "latency_seconds", row.get("latency_s", row.get("latency", None))
        )
        rows.append(row)

    rank = {role: index for index, role in enumerate(ROLE_ORDER)}
    rows.sort(key=lambda row: rank.get(str(row["role"]), len(rank)))
    return rows


def run_engine(engine: Any, waveform: np.ndarray, reference: str | None) -> list[dict[str, Any]]:
    started = time.perf_counter()
    try:
        payload = engine.run(
            waveform,
            reference=reference,
            noise_type="none",
            snr=None,
        )
    except TypeError:
        payload = engine.run(waveform, reference=reference)
    rows = normalise_results(payload)
    total = time.perf_counter() - started
    for row in rows:
        row.setdefault("total_demo_seconds", round(total, 4))
    return rows


def display_value(value: Any, *, percent: bool = False) -> str:
    if value is None or value == "":
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if percent:
        # The backend always returns aligned-v1 ratios. WER may legitimately
        # exceed 1.0 for a short utterance with many insertions, so magnitude
        # cannot distinguish a ratio from an already formatted percentage.
        numeric *= 100.0
        return f"{numeric:.4f}%"
    return f"{numeric:.4f}"


def result_table(rows: list[dict[str, Any]], has_reference: bool) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {
            "Model": row["model"],
            "Transcript": row["transcript"],
            "Latency (s)": display_value(row.get("latency_seconds")),
        }
        if has_reference:
            record.update({name.upper(): display_value(row.get(name), percent=True) for name in METRICS})
            record.update(
                {
                    "Sub": row.get("substitution", "—"),
                    "Del": row.get("deletion", "—"),
                    "Ins": row.get("insertion", "—"),
                    "Tone": row.get("tone", "—"),
                    "Diacritic": row.get("diacritic", "—"),
                    "Final coda": row.get("final_consonant", "—"),
                    "Short-word del": row.get("short_word_deletion", "—"),
                }
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def export_rows(
    rows: list[dict[str, Any]],
    *,
    reference: str | None,
    noise_type: str,
    snr: int | None,
    audio_info: Mapping[str, Any],
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for row in rows:
        flat = {
            "role": row.get("role"),
            "model": row.get("model"),
            "reference": reference or "",
            "transcript": row.get("transcript", ""),
            "noise_type": noise_type,
            "snr": "clean" if noise_type == "none" else snr,
            "sample_rate": SAMPLE_RATE,
            "duration_seconds": audio_info["processed_seconds"],
            "latency_seconds": row.get("latency_seconds"),
        }
        flat.update({name: row.get(name) for name in METRICS})
        flat.update({name: row.get(name) for name in ERROR_COUNTS})
        exported.append(flat)
    return exported


st.set_page_config(page_title="ViToneSR Demo", page_icon="🎙️", layout="wide")
st.title("🎙️ ViToneSR — Vietnamese ASR under noise")
st.caption(
    "So sánh cùng một audio giữa Ordinary LoRA, tone-aware lambda=0.05 và "
    "tone-aware lambda=0.1. Đây là demo định tính; kết luận paper dùng benchmark khóa."
)

with st.sidebar:
    st.header("Cấu hình demo")
    st.code(str(CONFIG_PATH.relative_to(ROOT)), language=None)
    st.write(f"Audio: mono, {SAMPLE_RATE:,} Hz, tối đa {MAX_SECONDS:g} giây")
    st.write("Decoding: greedy, Vietnamese transcription")
    st.info("Lần chạy đầu có thể chậm vì cần nạp backbone và ba LoRA adapter.")

left, right = st.columns([1, 1])
with left:
    st.subheader("1. Audio đầu vào")
    input_mode = st.radio(
        "Nguồn audio",
        ("Thu trực tiếp", "Tải WAV"),
        horizontal=True,
        label_visibility="collapsed",
    )
    if input_mode == "Thu trực tiếp":
        if hasattr(st, "audio_input"):
            # Streamlit records a browser-supported WAV rate; decode_audio
            # resamples it to the locked 16 kHz inference rate.
            source = st.audio_input("Bấm micro, đọc câu rồi dừng thu")
        else:
            st.warning("Streamlit hiện tại chưa hỗ trợ microphone. Hãy cập nhật hoặc tải WAV.")
            source = st.file_uploader("Tải audio thay thế", type=["wav"])
    else:
        source = st.file_uploader("Chọn file WAV", type=["wav"])

    reference = st.text_area(
        "Reference (không bắt buộc)",
        placeholder="Ví dụ: Bà bảo bé Bảo mang bốn quả bưởi về nhà.",
        help="Có reference để tính WER/CER/TER/DER/FCER/SWDR và Sub/Del/Ins.",
    ).strip()

with right:
    st.subheader("2. Điều kiện noise")
    noise_type = st.selectbox(
        "Noise",
        tuple(NOISE_LABELS),
        format_func=lambda key: NOISE_LABELS[key],
    )
    snr = st.select_slider(
        "SNR (dB)",
        options=list(CONFIG["noise"]["snr_db"]),
        value=10,
        disabled=noise_type == "none",
    )
    if noise_type != "none":
        st.caption("Noise tổng hợp chỉ dùng cho live demo, không thay thế MUSAN benchmark.")

clean_audio: np.ndarray | None = None
processed_audio: np.ndarray | None = None
audio_info: dict[str, Any] | None = None
if source is not None:
    try:
        clean_audio, audio_info = decode_audio(_audio_bytes(source))
        processed_audio = prepare_preview(
            clean_audio,
            noise_type,
            None if noise_type == "none" else int(snr),
        )
    except Exception as exc:  # Streamlit should show actionable input failures.
        st.error(f"Không đọc được audio: {exc}")

if processed_audio is not None and audio_info is not None:
    st.subheader("3. Nghe và kiểm tra audio")
    preview_left, preview_right = st.columns(2)
    with preview_left:
        st.caption("Audio clean/đã thu")
        st.audio(wav_bytes(clean_audio), format="audio/wav")
    with preview_right:
        st.caption(
            "Audio đưa vào model"
            if noise_type == "none"
            else f"Audio đưa vào model — {noise_type}, {snr} dB"
        )
        processed_wav = wav_bytes(processed_audio)
        st.audio(processed_wav, format="audio/wav")
        st.download_button(
            "Tải WAV đã xử lý",
            processed_wav,
            file_name=f"vitonesr_demo_{noise_type}_{'clean' if noise_type == 'none' else str(snr) + 'db'}.wav",
            mime="audio/wav",
        )
    if audio_info["clipped_to_max_seconds"]:
        st.warning(f"Audio dài hơn {MAX_SECONDS:g} giây và đã được cắt theo protocol.")
    st.caption(
        f"{audio_info['processed_seconds']:.3f} giây · {SAMPLE_RATE:,} Hz mono · "
        f"nguồn {audio_info['source_sample_rate']:,} Hz/{audio_info['source_channels']} kênh"
    )

run_clicked = st.button(
    "▶ Chạy cả 3 model",
    type="primary",
    disabled=processed_audio is None,
    use_container_width=True,
)

if run_clicked and processed_audio is not None and audio_info is not None:
    try:
        with st.spinner("Đang nạp checkpoint và chạy ba cấu hình..."):
            engine = load_engine()
            rows = run_engine(engine, processed_audio, reference or None)
        exports = export_rows(
            rows,
            reference=reference or None,
            noise_type=noise_type,
            snr=None if noise_type == "none" else int(snr),
            audio_info=audio_info,
        )
        st.session_state["demo_result"] = {
            "rows": rows,
            "exports": exports,
            "reference": reference,
            "noise_type": noise_type,
            "snr": None if noise_type == "none" else int(snr),
            "audio_info": audio_info,
        }
    except Exception as exc:
        st.exception(exc)

result_state = st.session_state.get("demo_result")
if result_state:
    st.subheader("4. Kết quả")
    rows = result_state["rows"]
    for row in rows:
        st.markdown(f"**{row['model']}**")
        st.write(row.get("transcript") or "*(Không có transcript)*")
        alignment = row.get("alignment") or []
        error_events = [event for event in alignment if event.get("operation") != "match"]
        if error_events:
            with st.expander(f"Chi tiết {len(error_events)} word-edit event"):
                event_rows = []
                for event in error_events:
                    event_rows.append(
                        {
                            "Operation": event.get("operation"),
                            "Reference": event.get("ref_token", ""),
                            "Hypothesis": event.get("hyp_token", ""),
                            "Tone error": bool(event.get("tone_error", False)),
                            "Diacritic error": bool(event.get("diacritic_error", False)),
                            "Final coda error": bool(
                                event.get("final_consonant_error", False)
                            ),
                            "Short-word deletion": bool(
                                event.get("short_word_deletion", False)
                            ),
                        }
                    )
                st.dataframe(event_rows, hide_index=True, use_container_width=True)

    table = result_table(rows, bool(result_state["reference"]))
    st.dataframe(table, hide_index=True, use_container_width=True)
    if not result_state["reference"]:
        st.info("Nhập reference rồi chạy lại để xem metric và phân rã lỗi.")

    exported = result_state["exports"]
    csv_data = pd.DataFrame.from_records(exported).to_csv(index=False).encode("utf-8-sig")
    json_payload = {
        "schema_version": "vitonesr_demo_result_v1",
        "qualitative_demo": True,
        "formal_artifacts_modified": False,
        "audio": result_state["audio_info"],
        "noise_type": result_state["noise_type"],
        "snr": result_state["snr"],
        "reference": result_state["reference"],
        "results": exported,
    }
    json_data = json.dumps(json_payload, ensure_ascii=False, indent=2).encode("utf-8")
    export_left, export_right = st.columns(2)
    with export_left:
        st.download_button(
            "Tải kết quả CSV",
            csv_data,
            file_name="vitonesr_demo_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_right:
        st.download_button(
            "Tải kết quả JSON",
            json_data,
            file_name="vitonesr_demo_results.json",
            mime="application/json",
            use_container_width=True,
        )

st.divider()
st.caption(
    "Demo không ghi vào outputs/paper_v2. Audio thực tế có SNR không biết; "
    "không dùng một câu demo để suy ra ý nghĩa thống kê."
)
