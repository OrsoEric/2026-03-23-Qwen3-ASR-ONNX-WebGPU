#!/usr/bin/env python3
"""
Qwen3-ASR ONNX Inference (Optimized + Correct Output)

Fixes:
- Restores correct prompt (audio start/end tokens)
- Restores decoding cleanup
- Restores language parsing
- Keeps I/O binding + performance optimizations
"""

#python qwen3-asr-onnx-v3.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav" "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU"

#python qwen3-asr-onnx-v3.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU"

import argparse
import time
from pathlib import Path
from typing import Optional

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

#C_S_EXECUTION_PROVIDER = "WebGpuExecutionProvider"

C_S_EXECUTION_PROVIDER = "CPUExecutionProvider"

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
    def __init__(self, path):
        self.tk = Tokenizer.from_file(path)

    def encode(self, t):
        return self.tk.encode(t).ids

    def decode(self, ids):
        return self.tk.decode(ids, skip_special_tokens=True)

# ── Pipeline ──────────────────────────────────────────

class Pipeline:

    def __init__(self, i_s_onnx_dir : str):

        st_onnx_options = ort.SessionOptions()
        st_onnx_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # profiling
        st_onnx_options.enable_profiling = C_X_ENABLE_PROFILER
        st_onnx_options.profile_file_prefix = "onnxruntime_profile"

        s_onnx_path_model = Path(i_s_onnx_dir) / "onnx_models"

        self.enc_conv = Pipeline._load_onnx( s_onnx_path_model / "encoder_conv.onnx", st_onnx_options )
        
        self.enc_tr = Pipeline._load_onnx( s_onnx_path_model / "encoder_transformer.onnx", st_onnx_options )
        
        self.dec_init = Pipeline._load_onnx( s_onnx_path_model / "decoder_init.int8.onnx", st_onnx_options )
        
        self.cl_onnx_dec_step = Pipeline._load_onnx( s_onnx_path_model / "decoder_step.int8.onnx", st_onnx_options )

        self.ann_embed = np.fromfile(
            str(s_onnx_path_model / "embed_tokens.bin"),
            dtype=np.float32
        ).reshape(VOCAB_SIZE, HIDDEN_SIZE)

        self.cl_tokenizer = SimpleTokenizer(str(Path(i_s_onnx_dir) / "tokenizer.json"))
        self.cl_mel_filters = get_mel_filters()

        self._attn_cache = None

    @staticmethod
    def _load_onnx(i_s_onnx_path : Path, i_st_onnx_options : ort.SessionOptions ) -> ort.InferenceSession:
        
        if i_s_onnx_path.exists() == False:
            raise Exception(f"cannot find {i_s_onnx_path}")
        else:
            n_size = i_s_onnx_path.stat().st_size
            print(f"Loading {i_s_onnx_path.stem} | ({ n_size / 1e6:.0f} MB)...")
            
        cl_onnx_model = ort.InferenceSession(
            str(i_s_onnx_path),
            i_st_onnx_options,
            providers=[C_S_EXECUTION_PROVIDER]
        )
        
        return cl_onnx_model
        
    # ── IO Binding ─────────────────────────────

    def _run(self, i_cl_session, i_aan_input):
        io = i_cl_session.io_binding()

        for k, v in i_aan_input.items():
            io.bind_ortvalue_input(k, ort.OrtValue.ortvalue_from_numpy(v))

        for o in i_cl_session.get_outputs():
            io.bind_output(o.name)

        i_cl_session.run_with_iobinding(io)
        return io.copy_outputs_to_cpu()

    # ── Encoder ────────────────────────────────

    def encode(self, mel):
        t0 = time.time()

        x = mel[np.newaxis, np.newaxis, :, :]
        conv = self._run(self.enc_conv, {"padded_mel_chunks": x})[0]

        hidden = conv[0]
        T = hidden.shape[0]

        if self._attn_cache is None or self._attn_cache.shape[-1] < T:
            self._attn_cache = np.zeros((1,1,T,T), dtype=np.float32)

        attn = self._attn_cache[:, :, :T, :T]

        out = self._run(self.enc_tr, {
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

        wav = load_audio(path)
        duration = len(wav) / SAMPLE_RATE

        # MEL
        t0 = time.time()
        mel = compute_mel(wav, self.cl_mel_filters)
        t_mel = time.time() - t0

        # ENCODER
        audio_feat, t_encoder = self.encode(mel)

        # PREP
        t0 = time.time()
        ids = self.build_prompt(len(audio_feat), language)
        emb = self.embed_inputs(ids, audio_feat)
        pos = np.arange(emb.shape[1]).reshape(1, -1).astype(np.int64)
        t_prepare = time.time() - t0

        # PREFILL
        t0 = time.time()
        logits, k, v = self._run(self.dec_init, {
            "input_embeds": emb,
            "position_ids": pos
        })
        t_prefill = time.time() - t0

        # DECODE
        t0 = time.time()

        next_token = int(np.argmax(logits[0, -1]))
        generated = [next_token]

        token_embed = np.empty((1,1,HIDDEN_SIZE), dtype=np.float32)
        pos_buf = np.empty((1,1), dtype=np.int64)

        cur = emb.shape[1]

        for _ in range(512):
            if next_token in (IM_END_ID, ENDOFTEXT_ID):
                break

            token_embed[0,0] = self.ann_embed[next_token]
            pos_buf[0,0] = cur

            logits, k, v = self._run(self.cl_onnx_dec_step, {
                "input_embeds": token_embed,
                "position_ids": pos_buf,
                "past_keys": k,
                "past_values": v
            })

            next_token = int(np.argmax(logits[0, -1]))
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
        rtf = total_time / duration

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
                "audio_duration_s": duration,
                "rtf": rtf,
                "tokens": tokens,
                "tps": tps,
            }
        }

    def save_profiles(self):
        for s in [self.enc_conv, self.enc_tr, self.dec_init, self.cl_onnx_dec_step]:
            print("Profile:", s.end_profiling())

# ── CLI ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio", nargs="+")
    p.add_argument("--onnx-dir", required=True)
    p.add_argument("--language", default=None)
    args = p.parse_args()

    pipe = Pipeline(args.onnx_dir)

    for f in args.audio:
        res = pipe.transcribe(f, args.language)
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
        pipe.save_profiles()

if __name__ == "__main__":
    main()