#!/usr/bin/env python3
"""
Qwen3-ASR ONNX Inference (Optimized + Correct Output + WebGPU KV on-device)

Fixes:
- Restores correct prompt (audio start/end tokens)
- Restores decoding cleanup
- Restores language parsing
- Keeps I/O binding + performance optimizations
- Keeps KV cache on WebGPU instead of copying back to CPU every step
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# ── Constants ─────────────────────────────────────────

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128

AUDIO_START_ID = 151669
AUDIO_END_ID = 151670
AUDIO_PAD_ID = 151676

IM_START_ID = 151644
IM_END_ID = 151645
ENDOFTEXT_ID = 151643
NEWLINE_ID = 198

VOCAB_SIZE = 151936
HIDDEN_SIZE = 1024

C_X_ENABLE_PROFILER = False
#C_S_EXECUTION_PROVIDER = "CPUExecutionProvider"
C_S_EXECUTION_PROVIDER = "WebGpuExecutionProvider"
C_N_MAX_TOKENS = 512

# ── Audio ─────────────────────────────────────────────

def load_audio(path):
    import librosa
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return wav.astype(np.float32)

def get_mel_filters():
    import librosa
    return librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS
    ).astype(np.float32)

def compute_mel(wav, mel_filters):
    import librosa
    stft = librosa.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft) ** 2
    mel = mel_filters @ mag
    log = np.log10(np.maximum(mel, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0).astype(np.float32)

# ── Tokenizer ─────────────────────────────────────────

class SimpleTokenizer:
    def __init__(self, i_s_path_tokenizer: Path):
        self.cl_tokenizer = Tokenizer.from_file(i_s_path_tokenizer)

    def encode(self, i_an_input):
        return self.cl_tokenizer.encode(i_an_input).ids

    def decode(self, i_an_tokens):
        return self.cl_tokenizer.decode(i_an_tokens, skip_special_tokens=True)

# ── Pipeline ──────────────────────────────────────────

class Pipeline:

    def __init__(self, i_s_onnx_dir: str):

        st_onnx_options = ort.SessionOptions()
        st_onnx_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        st_onnx_options.enable_profiling = C_X_ENABLE_PROFILER
        st_onnx_options.profile_file_prefix = "onnxruntime_profile"

        s_onnx_path_model = Path(i_s_onnx_dir) / "onnx_models"

        t_load_time: float = 0.0

        self.cl_onnx_encoder_convolution, t_elapsed = Pipeline._load_onnx(
            s_onnx_path_model / "encoder_conv.onnx", st_onnx_options
        )
        t_load_time += t_elapsed

        self.cl_onnx_encoder_transformer, t_elapsed = Pipeline._load_onnx(
            s_onnx_path_model / "encoder_transformer.onnx", st_onnx_options
        )
        t_load_time += t_elapsed

        self.cl_onnx_decoder_init, t_elapsed = Pipeline._load_onnx(
            s_onnx_path_model / "decoder_init.int8.onnx", st_onnx_options
        )
        t_load_time += t_elapsed

        self.cl_onnx_decoder_step, t_elapsed = Pipeline._load_onnx(
            s_onnx_path_model / "decoder_step.int8.onnx", st_onnx_options
        )
        t_load_time += t_elapsed

        t_now = time.time()
        self.ann_embed = np.fromfile(
            str(s_onnx_path_model / "embed_tokens.bin"),
            dtype=np.float32
        ).reshape(VOCAB_SIZE, HIDDEN_SIZE)
        t_elapsed = time.time() - t_now
        print(f"loaded embed tokens bin in {t_elapsed:.3f} [s]")
        t_load_time += t_elapsed

        t_now = time.time()
        self.cl_tokenizer = SimpleTokenizer(str(Path(i_s_onnx_dir) / "tokenizer.json"))
        self.cl_mel_filters = get_mel_filters()
        t_elapsed = time.time() - t_now
        print(f"loaded tokenizer {t_elapsed:.3f} [s]")
        t_load_time += t_elapsed

        print(f"LOADED ALL MODELS IN {t_load_time:.3f}")

        self._attn_cache = None

        # device tag for IOBinding (used to keep KV on WebGPU)
        if C_S_EXECUTION_PROVIDER == "WebGpuExecutionProvider":
            self.s_device_type = "webgpu"
            self.n_device_id = 0
        else:
            self.s_device_type = "cpu"
            self.n_device_id = 0

    @staticmethod
    def _load_onnx(i_s_onnx_path: Path, i_st_onnx_options: ort.SessionOptions) -> Tuple[ort.InferenceSession, float]:

        if i_s_onnx_path.exists() is False:
            raise Exception(f"cannot find {i_s_onnx_path}")

        n_size = i_s_onnx_path.stat().st_size
        if n_size < 1000000:
            s_onnx_path_data: Path = i_s_onnx_path.with_name(i_s_onnx_path.name + ".data")
            if s_onnx_path_data.exists() is True:
                n_size = s_onnx_path_data.stat().st_size

        print(f"Loading {i_s_onnx_path.stem} | ({n_size / 1e6:.0f} [MB])...")

        t_start = time.time()
        cl_onnx_model = ort.InferenceSession(
            str(i_s_onnx_path),
            i_st_onnx_options,
            providers=[C_S_EXECUTION_PROVIDER]
        )
        t_elapsed = time.time() - t_start
        print(f"Loaded in {t_elapsed:.3f} [s] ")

        return (cl_onnx_model, t_elapsed)

    # ── IO Binding ─────────────────────────────

    def _run(
        self,
        i_cl_session,
        i_aan_input,
        i_x_return_ortvalues: bool = False,
        i_x_outputs_on_device: bool = False,
    ):
        """
        i_x_return_ortvalues:
            - False: return numpy (CPU)
            - True:  return OrtValue objects
        i_x_outputs_on_device:
            - False: outputs are allocated on CPU
            - True:  outputs are allocated on self.s_device_type (e.g. WebGPU)
        """

        io = i_cl_session.io_binding()

        # bind inputs
        for k, v in i_aan_input.items():
            if isinstance(v, ort.OrtValue):
                io.bind_ortvalue_input(k, v)
            else:
                io.bind_ortvalue_input(k, ort.OrtValue.ortvalue_from_numpy(v))

        # bind outputs
        if i_x_outputs_on_device:
            # keep outputs (e.g. KV cache) on device (WebGPU)
            for o in i_cl_session.get_outputs():
                # let ORT allocate on the target device; shape/type inferred by runtime
                io.bind_output(
                    o.name,
                    self.s_device_type,
                    self.n_device_id,
                )
        else:
            # default: outputs on CPU
            for o in i_cl_session.get_outputs():
                io.bind_output(o.name)

        i_cl_session.run_with_iobinding(io)

        if i_x_return_ortvalues:
            # returns OrtValues; if outputs_on_device=True, they stay on device
            return io.get_outputs()
        else:
            # copies to CPU numpy
            return io.copy_outputs_to_cpu()

    # ── Encoder ────────────────────────────────

    def encode(self, mel):

        t0 = time.time()

        x = mel[np.newaxis, np.newaxis, :, :]
        conv = self._run(
            self.cl_onnx_encoder_convolution,
            {"padded_mel_chunks": x}
        )[0]

        hidden = conv[0]
        T = hidden.shape[0]

        if self._attn_cache is None or self._attn_cache.shape[-1] < T:
            self._attn_cache = np.zeros((1, 1, T, T), dtype=np.float32)

        attn = self._attn_cache[:, :, :T, :T]

        out = self._run(self.cl_onnx_encoder_transformer, {
            "hidden_states": hidden,
            "attention_mask": attn
        })[0]

        return out, time.time() - t0

    # ── Prompt ────────────────────────────────

    def build_prompt(self, n_audio, language: Optional[str] = None):

        ids = [IM_START_ID] + self.cl_tokenizer.encode("system") + [NEWLINE_ID, IM_END_ID, NEWLINE_ID]

        ids += [IM_START_ID] + self.cl_tokenizer.encode("user") + [NEWLINE_ID]

        ids += [AUDIO_START_ID]
        ids += [AUDIO_PAD_ID] * n_audio
        ids += [AUDIO_END_ID]

        ids += [IM_END_ID, NEWLINE_ID]

        ids += [IM_START_ID] + self.cl_tokenizer.encode("assistant") + [NEWLINE_ID]

        if language:
            ids += self.cl_tokenizer.encode(f"language {language}<asr_text>")

        return ids

    def embed_inputs(self, ids, audio):

        arr = np.array(ids)
        emb = self.ann_embed[arr]
        emb[arr == AUDIO_PAD_ID] = audio
        return emb[np.newaxis, :, :]

    # ── Transcribe ────────────────────────────

    def transcribe(self, path, language=None):

        total_start = time.time()

        an_wav = load_audio(path)
        n_duration_s = len(an_wav) / SAMPLE_RATE

        # MEL
        t0 = time.time()
        mel = compute_mel(an_wav, self.cl_mel_filters)
        t_mel = time.time() - t0

        # ENCODER
        audio_feat, t_encoder = self.encode(mel)

        # PREP
        t0 = time.time()
        ids = self.build_prompt(len(audio_feat), language)
        emb = self.embed_inputs(ids, audio_feat)
        pos = np.arange(emb.shape[1]).reshape(1, -1).astype(np.int64)
        t_prepare = time.time() - t0

        # PREFILL (KV stays on device for WebGPU)
        t0 = time.time()
        outputs = self._run(
            self.cl_onnx_decoder_init,
            {
                "input_embeds": emb,
                "position_ids": pos
            },
            i_x_return_ortvalues=True,
            i_x_outputs_on_device=(C_S_EXECUTION_PROVIDER == "WebGpuExecutionProvider"),
        )

        logits_ort = outputs[0]
        k = outputs[1]
        v = outputs[2]

        # only pull last token logits (reduces sync cost)
        last_logits = logits_ort.numpy()[:, -1, :]
        next_token = int(np.argmax(last_logits))

        t_prefill = time.time() - t0

        # DECODE
        t0 = time.time()

        generated = [next_token]

        token_embed = np.empty((1, 1, HIDDEN_SIZE), dtype=np.float32)
        pos_buf = np.empty((1, 1), dtype=np.int64)

        token_embed_ort = ort.OrtValue.ortvalue_from_numpy(token_embed)
        pos_buf_ort = ort.OrtValue.ortvalue_from_numpy(pos_buf)

        cur = emb.shape[1]

        for _ in range(C_N_MAX_TOKENS):
            if next_token in (IM_END_ID, ENDOFTEXT_ID):
                break

            token_embed[0, 0] = self.ann_embed[next_token]
            pos_buf[0, 0] = cur

            token_embed_ort.update_inplace(token_embed)
            pos_buf_ort.update_inplace(pos_buf)

            outputs = self._run(
                self.cl_onnx_decoder_step,
                {
                    "input_embeds": token_embed_ort,
                    "position_ids": pos_buf_ort,
                    "past_keys": k,
                    "past_values": v
                },
                i_x_return_ortvalues=True,
                i_x_outputs_on_device=(C_S_EXECUTION_PROVIDER == "WebGpuExecutionProvider"),
            )

            logits_ort = outputs[0]
            k = outputs[1]
            v = outputs[2]

            # pull only last token instead of full tensor
            last_logits = logits_ort.numpy()[:, -1, :]
            next_token = int(np.argmax(last_logits))

            generated.append(next_token)
            cur += 1

        t_decode = time.time() - t0

        # cleanup EOS
        if generated and generated[-1] in (IM_END_ID, ENDOFTEXT_ID):
            generated = generated[:-1]

        raw = self.cl_tokenizer.decode(generated)

        # parse language output
        parsed_lang = ""
        parsed_text = raw

        if "language " in raw and "<asr_text>" in raw:
            parts = raw.split("<asr_text>", 1)
            lang_part = parts[0]
            if lang_part.startswith("language "):
                parsed_lang = lang_part[len("language "):]
            parsed_text = parts[1] if len(parts) > 1 else ""

        total_time = time.time() - total_start

        tokens = len(generated)
        tps = tokens / t_decode if t_decode > 0 else 0
        rtf = total_time / n_duration_s

        return {
            "text": parsed_text.strip(),
            "language": parsed_lang,
            "timing": {
                "mel_s": t_mel,
                "encoder_s": t_encoder,
                "prepare_s": t_prepare,
                "prefill_s": t_prefill,
                "decode_s": t_decode,
                "total_s": total_time,
                "audio_duration_s": n_duration_s,
                "rtf": rtf,
                "tokens": tokens,
                "tps": tps,
            }
        }

    def save_profiles(self):
        for s in [
            self.cl_onnx_encoder_convolution,
            self.cl_onnx_encoder_transformer,
            self.cl_onnx_decoder_init,
            self.cl_onnx_decoder_step,
        ]:
            print("Profile:", s.end_profiling())

# ── CLI ─────────────────────────────────────────────

def main():

    p = argparse.ArgumentParser()
    p.add_argument("audio", nargs="+")
    p.add_argument("--onnx-dir", required=True)
    p.add_argument("--language", default=None)
    args = p.parse_args()

    cl_pipeline = Pipeline(args.onnx_dir)

    for f in args.audio:
        res = cl_pipeline.transcribe(f, args.language)
        t = res["timing"]

        print(f"\n[{f}] ({t['audio_duration_s']:.1f}s, RTF {t['rtf']:.2f})")

        if res["language"]:
            print("Language:", res["language"])

        print(res["text"])

        print(
            f"mel {t['mel_s']:.3f}s | "
            f"enc {t['encoder_s']:.3f}s | "
            f"prefill {t['prefill_s']:.3f}s | "
            f"decode {t['decode_s']:.3f}s | "
            f"tokens {t['tokens']} | "
            f"tps {t['tps']:.1f}"
        )

    if C_X_ENABLE_PROFILER:
        cl_pipeline.save_profiles()

if __name__ == "__main__":
    main()
