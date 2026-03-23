# ISSUE

[GITHUB ISSUE](https://github.com/microsoft/onnxruntime/issues/27809)

# USECASE

https://huggingface.co/Daumee/Qwen3-ASR-0.6B-ONNX-CPU

I'm looking at ONNX for local inference in robots. I tested on several devices with poor results, right now I'm testing on Intel Core 100 series and AMD 7640u. AMD is unable to provide any GPU acceleration to any ML runtime on such APU.

Compared to llama.cpp/Vulkan, ONNX is attractive because of the much easier multimodal capability, in embedded this allows to ingest audio, text, images and video much easier than using pytorch and all it's dependencies.

# PROBLEM llama.cpp/Vulkan twice as fast as Qwen 3 0.6B/WebGPU

There is an enormous performance penality in using ONNX compared to llama.cpp/Vulkan.

[In tests using Qwen 3 0.6B on 7640u AMD APU with 32GB of DDR5-5600](https://github.com/microsoft/onnxruntime/issues/21917#issuecomment-3964767328) ONNX WebGPU EP (36.51 t/s) is slightly faster than ONNX CPU EP (31.64 t/s). llama.cpp/Vulkan (59.35t/s) CPU (40.34t/s)

llama.cpp/Vulkan is around twice as fast as ONNX/WebGPU


# PROBLEM Qwen 3 AST 0.6B ONNX CPU four times as fast as ONNX WeBGPU

I got around to test an audio ASR model Qwen 3 ASR 0.6B for the multimodal audio capability, and there seems to be an extreme performance penality to WebGPU compared to CPU EP.

I can't easily test llama.cpp/Vulkan audio for comparison, but I would expect it to be twice as fast as the 


# Code

I started from the official repo, implemented profiling, loading of models and io binding performance improvements.



<details>
<summary>CODE</summary>

```cmd
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

C_S_EXECUTION_PROVIDER = "WebGpuExecutionProvider"

#C_S_EXECUTION_PROVIDER = "CPUExecutionProvider"

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

    def __init__(self, onnx_dir):

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # profiling
        opts.enable_profiling = C_X_ENABLE_PROFILER
        opts.profile_file_prefix = "onnxruntime_profile"

        base = Path(onnx_dir) / "onnx_models"

        self.enc_conv = ort.InferenceSession(
            str(base / "encoder_conv.onnx"),
            opts,
            providers=[C_S_EXECUTION_PROVIDER]
        )

        self.enc_tr = ort.InferenceSession(
            str(base / "encoder_transformer.onnx"),
            opts,
            providers=[C_S_EXECUTION_PROVIDER]
        )

        self.dec_init = ort.InferenceSession(
            str(base / "decoder_init.int8.onnx"),
            opts,
            providers=[C_S_EXECUTION_PROVIDER]
        )

        self.dec_step = ort.InferenceSession(
            str(base / "decoder_step.int8.onnx"),
            opts,
            providers=[C_S_EXECUTION_PROVIDER]
        )

        self.embed = np.fromfile(
            str(base / "embed_tokens.bin"),
            dtype=np.float32
        ).reshape(VOCAB_SIZE, HIDDEN_SIZE)

        self.tk = SimpleTokenizer(str(Path(onnx_dir) / "tokenizer.json"))
        self.mel_filters = get_mel_filters()

        self._attn_cache = None

    # ── IO Binding ─────────────────────────────

    def _run(self, sess, inputs):
        io = sess.io_binding()

        for k, v in inputs.items():
            io.bind_ortvalue_input(k, ort.OrtValue.ortvalue_from_numpy(v))

        for o in sess.get_outputs():
            io.bind_output(o.name)

        sess.run_with_iobinding(io)
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
        ids = [IM_START_ID] + self.tk.encode("system") + [NEWLINE_ID, IM_END_ID, NEWLINE_ID]

        ids += [IM_START_ID] + self.tk.encode("user") + [NEWLINE_ID]

        ids += [AUDIO_START_ID]
        ids += [AUDIO_PAD_ID] * n_audio
        ids += [AUDIO_END_ID]

        ids += [IM_END_ID, NEWLINE_ID]

        ids += [IM_START_ID] + self.tk.encode("assistant") + [NEWLINE_ID]

        if language:
            ids += self.tk.encode(f"language {language}<asr_text>")

        return ids

    def embed_inputs(self, ids, audio):
        arr = np.array(ids)
        emb = self.embed[arr]
        emb[arr == AUDIO_PAD_ID] = audio
        return emb[np.newaxis, :, :]

    # ── Transcribe ────────────────────────────

    def transcribe(self, path, language=None):

        total_start = time.time()

        wav = load_audio(path)
        duration = len(wav) / SAMPLE_RATE

        # MEL
        t0 = time.time()
        mel = compute_mel(wav, self.mel_filters)
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

            token_embed[0,0] = self.embed[next_token]
            pos_buf[0,0] = cur

            logits, k, v = self._run(self.dec_step, {
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

        raw = self.tk.decode(generated)

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
        for s in [self.enc_conv, self.enc_tr, self.dec_init, self.dec_step]:
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
```

</details>

Command line call, i provide the folder where the onnx root is, and two sample waw audio files

```cmd
python qwen3-asr-onnx-v3.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav" "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU"
```

## CPU EP Performance 22 to 34 TPS


<details>
<summary>Qwen 3 4B</summary>

```
(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>python qwen3-asr-onnx-v3.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav" "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU"

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav] (18.0s, RTF 0.19)
Language: None
I was entrusted this for thee by Thomas Forma Master, 'cause a bell of calling forth spirits summoned thee from us in the city. This bell of calling forth spirits. A bell of calling forth.
mel 0.007s | enc 0.379s | prefill 0.346s | decode 1.984s | tokens 44 | tps 22.2

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav] (7.4s, RTF 0.18)
Language: English
Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. "It is certainly very like the old portrait."     
mel 0.002s | enc 0.186s | prefill 0.113s | decode 1.022s | tokens 35 | tps 34.3
```

</details>

## WebGPU EP Performance 7 to 9TPS

<details>
<summary>PERFORMANCE DETAILS</summary>

```
(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>python qwen3-asr-onnx-v3.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav" "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU"
2026-03-23 15:23:39.9483622 [W:onnxruntime:, session_state.cc:1327 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.    
2026-03-23 15:23:39.9528895 [W:onnxruntime:, session_state.cc:1329 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
2026-03-23 15:23:40.0724043 [W:onnxruntime:, session_state.cc:1327 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.    
2026-03-23 15:23:40.0771509 [W:onnxruntime:, session_state.cc:1329 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
2026-03-23 15:23:41.1673083 [W:onnxruntime:, session_state.cc:1327 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.    
2026-03-23 15:23:41.1723952 [W:onnxruntime:, session_state.cc:1329 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
2026-03-23 15:23:42.0740586 [W:onnxruntime:, session_state.cc:1327 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.    
2026-03-23 15:23:42.0792135 [W:onnxruntime:, session_state.cc:1329 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav] (18.0s, RTF 0.59)
Language: None
I was entrusted this for thee by Thomas Former Master, 'cause a bell of calling forth, 'tis summoned thee from us in the city, 'tis for a beggar man, a beggar, a little poor poor bastard.
mel 0.007s | enc 0.793s | prefill 1.151s | decode 8.087s | tokens 53 | tps 6.6

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav] (7.4s, RTF 0.60)
Language: English
Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. It is certainly very like the old portrait.       
mel 0.002s | enc 0.264s | prefill 0.386s | decode 3.776s | tokens 34 | tps 9.0
```

</details>

# CONCLUSIONS

Performance of Qwen 3 ASR 0.6B should be around 70 TPS on AMD 7640u.

In my LLM tests WebGPU EP 35TPS has some enormous performance penality compared to llama.cpp/Vulkan 70TPS, halving performance.

In my Audio LLM tests, WebGPU EP (9TPS) is enormously slower than the CPU EP (34TPS), down to quarter performance.

It might have something to do with low performance Direct3D fallback instead of Vulkan.

It might have something to do with AMD listing 512MB dedicated memory, and WebGPU/Direct3D being unable to use the much larger 32GB/16GB shared memory, while llama.cpp/Vulkan seems to have no issue using with the shared memory.



