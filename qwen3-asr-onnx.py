import os
import time
import numpy as np
import librosa
import onnxruntime as ort
from transformers import AutoProcessor

# ======================
# CONFIG
# ======================
model_path = r"D:\ASR_Models\Qwen3-ASR-0.6B-ONNX"
onnx_path = os.path.join(model_path, "model.onnx")

AUDIO_PATH = "audio.wav"
SAMPLE_RATE = 16000

# ======================
# LOAD PROCESSOR
# ======================
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

# ======================
# LOAD AUDIO
# ======================
audio, sr = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE)
audio = audio.astype(np.float32)

# ======================
# FEATURE EXTRACTION
# ======================
inputs = processor(
    audio,
    sampling_rate=SAMPLE_RATE,
    return_tensors="np"
)

input_features = inputs["input_features"]

# ======================
# ONNX SESSION (WebGPU)
# ======================
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True
sess_options.log_severity_level = 3

session = ort.InferenceSession(
    onnx_path,
    providers=["WebGpuExecutionProvider"],
    sess_options=sess_options
)

# ======================
# PROFILING
# ======================
print("\n--- Starting ASR Inference ---")

start_time = time.perf_counter()

# Run full sequence (no token loop like LLM)
outputs = session.run(None, {
    "input_features": input_features
})

end_time = time.perf_counter()

logits = outputs[0]

# ======================
# DECODE
# ======================
predicted_ids = np.argmax(logits, axis=-1)

transcription = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True
)[0]

# ======================
# REPORT
# ======================
total_time = end_time - start_time
audio_duration = len(audio) / SAMPLE_RATE

print("\n=== TRANSCRIPTION ===")
print(transcription)

print("\n" + "="*30)
print("       ASR PROFILE")
print("="*30)
print(f"Audio Length:      {audio_duration:.2f} s")
print(f"Inference Time:    {total_time:.4f} s")
print(f"Real-time factor:  {total_time / audio_duration:.3f}x")
print("="*30)