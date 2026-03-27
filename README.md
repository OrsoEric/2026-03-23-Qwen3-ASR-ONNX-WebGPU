# 

```cmd
Microsoft Windows [Version 10.0.26100.6584]
(c) Microsoft Corporation. All rights reserved.

D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR\New folder>git clone https://huggingface.co/Daumee/Qwen3-ASR-0.6B-ONNX-CPU
Cloning into 'Qwen3-ASR-0.6B-ONNX-CPU'...
remote: Enumerating objects: 135, done.
remote: Counting objects: 100% (131/131), done.
remote: Compressing objects: 100% (131/131), done.
remote: Total 135 (delta 59), reused 0 (delta 0), pack-reused 4 (from 1)
Receiving objects: 100% (135/135), 82.13 KiB | 801.00 KiB/s, done.
Resolving deltas: 100% (59/59), done.
Updating files: 100% (15/15), done.
Filtering content: 100% (11/11), 2.40 GiB | 10.83 MiB/s, done.
```

# ENVIRONMENT

```cmd
uv venv .venv --python 3.13

call .venv\Scripts\activate.bat

uv init

uv pip install onnxruntime-webgpu --link-mode=copy

uv pip install librosa soundfile tokenizers --link-mode=copy

uv pip freeze > requirements.txt
```





<details>
<summary>Install dependencies create UV</summary>

```
D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>uv venv .venv --python 3.13
Using CPython 3.13.1 interpreter at: D:\Programs\Python_3_13\python.exe
Creating virtual environment at:
 .venv
Activate with: .venv\Scripts\activate

D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>call .venv\Scripts\activate.bat

(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>uv init
Initialized project `2026-03-23-onnx-qwen3-asr`

(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>uv pip install onnxruntime-webgpu --link-mode=copy   
Resolved 7 packages in 447ms    
Prepared 2 packages in 1.84s    
Installed 7 packages in 1.40s   
 + flatbuffers==25.12.19        
 + mpmath==1.3.0                
 + numpy==2.4.3
 + onnxruntime-webgpu==1.25.0.dev20260212001
 + packaging==26.0
 + protobuf==7.34.1
 + sympy==1.14.0


(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>uv pip install librosa soundfile tokenizers --link-mode=copy
Resolved 49 packages in 523ms   
Prepared 16 packages in 10.15s  
Installed 47 packages in 1.63s  
 + annotated-doc==0.0.4         
 + anyio==4.12.1                
 + audioop-lts==0.2.2           
 + audioread==3.1.0             
 + certifi==2026.2.25           
 + cffi==2.0.0                  
 + charset-normalizer==3.4.6    
 + click==8.3.1                 
 + colorama==0.4.6              
 + decorator==5.2.1
 + filelock==3.25.2             
 + fsspec==2026.2.0             
 + h11==0.16.0                  
 + hf-xet==1.4.2                
 + httpcore==1.0.9              
 + httpx==0.28.1                
 + huggingface-hub==1.7.2
 + idna==3.11                   
 + joblib==1.5.3                
 + lazy-loader==0.5
 + librosa==0.11.0
 + llvmlite==0.46.0
 + markdown-it-py==4.0.0        
 + mdurl==0.1.2
 + msgpack==1.1.2
 + numba==0.64.0
 + platformdirs==4.9.4
 + pooch==1.9.0
 + pycparser==3.0
 + pygments==2.19.2
 + pyyaml==6.0.3
 + requests==2.32.5
 + rich==14.3.3
 + scikit-learn==1.8.0
 + scipy==1.17.1
 + shellingham==1.5.4
 + soundfile==0.13.1
 + soxr==1.0.0
 + standard-aifc==3.13.0        
 + standard-chunk==3.13.0
 + standard-sunau==3.13.0       
 + threadpoolctl==3.6.0
 + tokenizers==0.22.2
 + tqdm==4.67.3
 + typer==0.24.1
 + typing-extensions==4.15.0    
 + urllib3==2.6.3

(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>uv pip install onnxruntime-webgpu --link-mode=copy   
Resolved 7 packages in 447ms    
Prepared 2 packages in 1.84s    
Installed 7 packages in 1.40s   
 + flatbuffers==25.12.19        
 + mpmath==1.3.0                
 + numpy==2.4.3
 + onnxruntime-webgpu==1.25.0.dev20260212001
 + packaging==26.0
 + protobuf==7.34.1
 + sympy==1.14.0

(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>python onnx_inference.py
usage: onnx_inference.py
       [-h]
       [--language LANGUAGE]
       [--onnx-dir ONNX_DIR]
       [--max-new-tokens MAX_NEW_TOKENS]
       [--quantize {none,int8}]
       [--chunk-sec CHUNK_SEC]
       [--threads THREADS]
       [--json]
       audio [audio ...]        
onnx_inference.py: error: the following arguments are required: audio

(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>uv pip install librosa soundfile tokenizers --link-mode=copy
Resolved 49 packages in 523ms   
Prepared 16 packages in 10.15s  
Installed 47 packages in 1.63s  
 + annotated-doc==0.0.4         
 + anyio==4.12.1                
 + audioop-lts==0.2.2           
 + audioread==3.1.0             
 + certifi==2026.2.25           
 + cffi==2.0.0                  
 + charset-normalizer==3.4.6    
 + click==8.3.1                 
 + colorama==0.4.6              
 + decorator==5.2.1
 + filelock==3.25.2             
 + fsspec==2026.2.0             
 + h11==0.16.0                  
 + hf-xet==1.4.2                
 + httpcore==1.0.9              
 + httpx==0.28.1                
 + huggingface-hub==1.7.2
 + idna==3.11                   
 + joblib==1.5.3                
 + lazy-loader==0.5
 + librosa==0.11.0
 + llvmlite==0.46.0
 + markdown-it-py==4.0.0        
 + mdurl==0.1.2
 + msgpack==1.1.2
 + numba==0.64.0
 + platformdirs==4.9.4
 + pooch==1.9.0
 + pycparser==3.0
 + pygments==2.19.2
 + pyyaml==6.0.3
 + requests==2.32.5
 + rich==14.3.3
 + scikit-learn==1.8.0
 + scipy==1.17.1
 + shellingham==1.5.4
 + soundfile==0.13.1
 + soxr==1.0.0
 + standard-aifc==3.13.0        
 + standard-chunk==3.13.0
 + standard-sunau==3.13.0       
 + threadpoolctl==3.6.0
 + tokenizers==0.22.2
 + tqdm==4.67.3
 + typer==0.24.1
 + typing-extensions==4.15.0    
 + urllib3==2.6.3

(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>uv pip freeze > requirements.txt
```
</details>

# PATH

MODEL

D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU

AUDIO SAMPLE 

D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav

python onnx_inference.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav"

python onnx_inference.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav" --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU\onnx_models"

# V2

it's incompetently designed. it redownloads the tokenizer...

python qwen3-asr-onnx-v2.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav" --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU"

# quantization

python qwen3-asr-onnx-v2.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav" --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU" --quantize int8

# working

```python qwen3-asr-onnx-v2.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav" --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU" --quantize int8```

```cmd
(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>python qwen3-asr-onnx-v2.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav" --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU" --quantize int8
Loading ONNX models (decoder: INT8)...
Loading embeddings (622 MB)...
Looking for tokenizer at: D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU\tokenizer.json
Pipeline ready.

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav] (7.4s, RTF 1.75x)
  Language: English
  Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. "It is certainly very like the old portrait." 
  Encoder: 0.191s | Prefill: 0.121s | Decode: 0.928s | Tokens: 35
```

# CPU EXECUTION

<details>
<summary>CMD LINE</summary>

```
```

</details>



# WEBGPU EXECUTION


<details>
<summary>CMD LINE</summary>

```
(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>python qwen3-asr-onnx-v2.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav" --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU" --quantize int8
Loading ONNX models (decoder: INT8)...
Loading embeddings (622 MB)...
Loading tokenizer at: D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU\tokenizer.json | Size: (11 MB)...
Pipeline ready.

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav] (7.4s, RTF 0.89x)
  Language: English
  Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. "It is certainly very like the old portrait." 
  Encoder: 0.992s | Prefill: 0.921s | Decode: 4.063s | Tokens: 35

(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>python qwen3-asr-onnx-v2.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav" --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU" --quantize int8
Loading ONNX models (decoder: INT8)...
Loading embeddings (622 MB)...
Loading tokenizer at: D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU\tokenizer.json | Size: (11 MB)...
Pipeline ready.

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav] (7.4s, RTF 0.77x)
  Language: English
  Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. "It is certainly very like the old portrait." 
  Encoder: 0.569s | Prefill: 0.621s | Decode: 3.851s | Tokens: 35
```

</details>

#



python qwen3-asr-onnx-v2.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav" "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU" --quantize int8



# Test two audios 

## CPU

<details>
<summary>Qwen 3 4B</summary>

```
python qwen3-asr-onnxx-v2.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav" "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU" --quantize int8
Loading ONNX models (decoder: INT8)...
Loading encoder_conv (0 MB)...
Loading encoder_transformer (0 MB)...
Loading decoder_init (598 MB)...
Loading decoder_step (598 MB)...
Loading embeddings (622 MB)...
Loading tokenizer at: D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU\tokenizer.json | Size: (11 MB)...
Pipeline ready.

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav] (18.0s, RTF 0.22x)
  Language: English
  I was entrusted this for thee by Torrent's former master. Tis a bell for calling forth spirits. Summon them with it. From ash and return to the earth tree, the spirits will obey thine command but briefly, as they recall battles past.
  Encoder: 0.467s | Prefill: 0.316s | Decode: 1.855s | Tokens: 54

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav] (7.4s, RTF 0.16x)
  Language: English
  Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. "It is certainly very like the old portrait." 
  Encoder: 0.175s | Prefill: 0.120s | Decode: 0.868s | Tokens: 35

(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>python qwen3-asr-onnx-v2.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav" "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU" --quantize int8
Loading ONNX models (decoder: INT8)...
Loading encoder_conv (0 MB)...
Loading encoder_transformer (0 MB)...
Loading decoder_init (598 MB)...
Loading decoder_step (598 MB)...
Loading embeddings (622 MB)...
Loading tokenizer at: D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU\tokenizer.json | Size: (11 MB)...
Pipeline ready.

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav] (18.0s, RTF 0.19x)
  Language: English
  I was entrusted this for thee by Torrent's former master. Tis a bell for calling forth spirits. Summon them with it. From ash and return to the earth tree, the spirits will obey thine command but briefly, as they recall battles past.
  Encoder: 0.512s | Prefill: 0.343s | Decode: 1.929s | Tokens: 54

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav] (7.4s, RTF 0.16x)
  Language: English
  Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. "It is certainly very like the old portrait." 
  Encoder: 0.200s | Prefill: 0.128s | Decode: 0.889s | Tokens: 35
```

</details>

# WEBGPU EP

<details>
<summary>WebGPU EP Details</summary>

```
(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>python qwen3-asr-onnx-v2.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav" "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU" --quantize int8
Loading ONNX models (decoder: INT8)...
Loading encoder_conv (0 MB)...
Loading encoder_transformer (0 MB)...
Loading decoder_init (598 MB)...
Loading decoder_step (598 MB)...
Loading embeddings (622 MB)...
Loading tokenizer at: D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU\tokenizer.json | Size: (11 MB)...
Pipeline ready.

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav] (18.0s, RTF 0.61x)
  Language: English
  I was entrusted this for thee by Torrent's former master. Tis a bell for calling forth spirits. Summon them with it. From ash and return to the earth tree, the spirits will obey thine command, but briefly, as they recall battles past.
  Encoder: 0.935s | Prefill: 1.227s | Decode: 8.208s | Tokens: 55

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav] (7.4s, RTF 0.57x)
  Language: English
  Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. "It is certainly very like the old portrait." 
  Encoder: 0.356s | Prefill: 0.284s | Decode: 3.595s | Tokens: 35

(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>python qwen3-asr-onnx-v2.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav" "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU" --quantize int8
Loading ONNX models (decoder: INT8)...
Loading encoder_conv (0 MB)...
Loading encoder_transformer (0 MB)...
Loading decoder_init (598 MB)...
Loading decoder_step (598 MB)...
Loading embeddings (622 MB)...
Loading tokenizer at: D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU\tokenizer.json | Size: (11 MB)...
Pipeline ready.

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav] (18.0s, RTF 0.60x)
  Language: English
  I was entrusted this for thee by Torrent's former master. Tis a bell for calling forth spirits. Summon them with it. From ash and return to the earth tree, the spirits will obey thine command, but briefly, as they recall battles past.
  Encoder: 0.842s | Prefill: 1.037s | Decode: 8.239s | Tokens: 55

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav] (7.4s, RTF 0.58x)
  Language: English
  Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. "It is certainly very like the old portrait." 
  Encoder: 0.304s | Prefill: 0.301s | Decode: 3.722s | Tokens: 35
```

</details>



# 





Performance is 26 to 36 TPS


<details>
<summary>ONNX CPU EP Details</summary>

```

python qwen3-asr-onnx-v2.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav" "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU" --quantize int8
Loading ONNX models (decoder: INT8)...
Loading encoder_conv (0 MB)...
Loading encoder_transformer (0 MB)...
Loading decoder_init (598 MB)...
Loading decoder_step (598 MB)...
Loading embeddings (622 MB)...
Loading tokenizer at: D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU\tokenizer.json | Size: (11 MB)...
Pipeline ready.

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav] (18.0s, RTF 0.19x)
  Language: English
  I was entrusted this for thee by Torrent's former master. Tis a bell for calling forth spirits. Summon them with it. From ash and return to the earth tree, the spirits will obey thine command but briefly, as they recall battles past.
  Encoder: 0.556s | Prefill: 0.335s | Decode: 1.969s | Tokens: 54 | TPS 27.429026500959928      

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav] (7.4s, RTF 0.16x)
  Language: English
  Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. "It is certainly very like the old portrait." 
  Encoder: 0.191s | Prefill: 0.126s | Decode: 0.875s | Tokens: 35 | TPS 39.97879063596229       

(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>python qwen3-asr-onnx-v2.py "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav" "D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU" --quantize int8
Loading ONNX models (decoder: INT8)...
Loading encoder_conv (0 MB)...
Loading encoder_transformer (0 MB)...
Loading decoder_init (598 MB)...
Loading decoder_step (598 MB)...
Loading embeddings (622 MB)...
Loading tokenizer at: D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU\tokenizer.json | Size: (11 MB)...
Pipeline ready.

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-Ranni-18s.wav] (18.0s, RTF 0.20x)
  Language: English
  I was entrusted this for thee by Torrent's former master. Tis a bell for calling forth spirits. Summon them with it. From ash and return to the earth tree, the spirits will obey thine command but briefly, as they recall battles past.
  Encoder: 0.508s | Prefill: 0.350s | Decode: 2.042s | Tokens: 54 | TPS 26.444238383935563      

[D:\Data\Project\Project-LLM\Audio Samples\SAMPLE-british-man-7s.wav] (7.4s, RTF 0.17x)
  Language: English
  Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. "It is certainly very like the old portrait." 
  Encoder: 0.185s | Prefill: 0.138s | Decode: 0.967s | Tokens: 35 | TPS 36.21095006462689  

```

</details>



# V4 Add IO binding to KV cache (FAIL)


```cmd
python qwen3-asr-onnx-v4.py "D:\Data\Project\Project-LLM\Audio Samples\wav\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU"
```


<details>
<summary>CPU</summary>

```
(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>python qwen3-asr-onnx-v4.py "D:\Data\Project\Project-LLM\Audio Samples\wav\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU" 
Loading encoder_conv | (50 [MB])...
Loaded in 0.085 [s] 
Loading encoder_transformer | (701 [MB])...
Loaded in 0.883 [s] 
Loading decoder_init.int8 | (598 [MB])...
Loaded in 1.388 [s] 
Loading decoder_step.int8 | (598 [MB])...
Loaded in 1.371 [s] 
loaded embed tokens bin in 0.169 [s]
loaded tokenizer 2.064 [s]
LOADED ALL MODELS IN 5.961

[D:\Data\Project\Project-LLM\Audio Samples\wav\SAMPLE-british-man-7s.wav] (7.4s, RTF 0.56)
Language: English
Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. "It is certainly very like the old portrait."     
mel 0.009s | enc 0.417s | prefill 0.227s | decode 1.411s | tokens 35 | tps 24.8

```

</details>

<details>
<summary>GPU</summary>

```
(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>python qwen3-asr-onnx-v4.py "D:\Data\Project\Project-LLM\Audio Samples\wav\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU" 
Loading encoder_conv | (50 [MB])...
2026-03-27 07:25:42.9787669 [W:onnxruntime:, session_state.cc:1327 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.    
2026-03-27 07:25:42.9899961 [W:onnxruntime:, session_state.cc:1329 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
Loaded in 0.874 [s] 
Loading encoder_transformer | (701 [MB])...
2026-03-27 07:25:43.3311327 [W:onnxruntime:, session_state.cc:1327 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.    
2026-03-27 07:25:43.3401527 [W:onnxruntime:, session_state.cc:1329 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
Loaded in 1.053 [s] 
Loading decoder_init.int8 | (598 [MB])...
2026-03-27 07:25:45.1252285 [W:onnxruntime:, session_state.cc:1327 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.    
2026-03-27 07:25:45.1355850 [W:onnxruntime:, session_state.cc:1329 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
Loaded in 1.555 [s] 
Loading decoder_step.int8 | (598 [MB])...
2026-03-27 07:25:46.6343121 [W:onnxruntime:, session_state.cc:1327 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.    
2026-03-27 07:25:46.6506720 [W:onnxruntime:, session_state.cc:1329 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
Loaded in 1.494 [s] 
loaded embed tokens bin in 0.175 [s]
loaded tokenizer 2.115 [s]
LOADED ALL MODELS IN 7.265

[D:\Data\Project\Project-LLM\Audio Samples\wav\SAMPLE-british-man-7s.wav] (7.4s, RTF 1.17)
Language: English
Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. It is certainly very like the old portrait.       
mel 0.008s | enc 0.804s | prefill 1.079s | decode 5.557s | tokens 34 | tps 6.1

```

</details>

# V5 

patch some of the KV cache

```cmd
python qwen3-asr-onnx-v5.py "D:\Data\Project\Project-LLM\Audio Samples\wav\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU"
```

# V6

GPT gets confused

patch the onnx models

```cmd
python qwen3-asr-onnx-v6.py "D:\Data\Project\Project-LLM\Audio Samples\wav\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU"
```

# V7

bingchat

```cmd
python qwen3-asr-onnx-v7.py "D:\Data\Project\Project-LLM\Audio Samples\wav\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU"
```


```cmd
(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>python qwen3-asr-onnx-v7.py "D:\Data\Project\Project-LLM\Audio Samples\wav\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU"
Loading encoder_conv | (50 [MB])...
Loaded in 0.048 [s] 
Loading encoder_transformer | (701 [MB])...
Loaded in 0.524 [s] 
Loading decoder_init.int8 | (598 [MB])...
Loaded in 0.832 [s] 
Loading decoder_step.int8 | (598 [MB])...
Loaded in 0.946 [s] 
loaded embed tokens bin in 0.176 [s]
loaded tokenizer 1.163 [s]
LOADED ALL MODELS IN 3.690

[D:\Data\Project\Project-LLM\Audio Samples\wav\SAMPLE-british-man-7s.wav] (7.4s, RTF 0.31)      
Language: English
Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. "It is certainly very like the old portrait."   
mel 0.006s | enc 0.313s | prefill 0.206s | decode 1.128s | tokens 35 | tps 31.0

(.venv) D:\Data\Project\Project-LLM\Runtime ONNX\2026-03-23 ONNX Qwen3 ASR>python qwen3-asr-onnx-v7.py "D:\Data\Project\Project-LLM\Audio Samples\wav\SAMPLE-british-man-7s.wav"  --onnx-dir "D:\LLM_Models\Qwen\Qwen3-ASR-0.6B-ONNX-CPU"
Loading encoder_conv | (50 [MB])...
2026-03-27 12:14:52.4418538 [W:onnxruntime:, session_state.cc:1327 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
2026-03-27 12:14:52.4473450 [W:onnxruntime:, session_state.cc:1329 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
Loaded in 0.192 [s] 
Loading encoder_transformer | (701 [MB])...
2026-03-27 12:14:52.5584098 [W:onnxruntime:, session_state.cc:1327 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
2026-03-27 12:14:52.5635922 [W:onnxruntime:, session_state.cc:1329 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
Loaded in 0.702 [s] 
Loading decoder_init.int8 | (598 [MB])...
2026-03-27 12:14:53.7359630 [W:onnxruntime:, session_state.cc:1327 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
2026-03-27 12:14:53.7411025 [W:onnxruntime:, session_state.cc:1329 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
Loaded in 0.894 [s] 
Loading decoder_step.int8 | (598 [MB])...
2026-03-27 12:14:54.6294418 [W:onnxruntime:, session_state.cc:1327 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
2026-03-27 12:14:54.6343810 [W:onnxruntime:, session_state.cc:1329 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
Loaded in 0.919 [s] 
loaded embed tokens bin in 0.147 [s]
loaded tokenizer 1.136 [s]
LOADED ALL MODELS IN 3.991

[D:\Data\Project\Project-LLM\Audio Samples\wav\SAMPLE-british-man-7s.wav] (7.4s, RTF 0.83)      
Language: English
Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. It is certainly very like the old portrait.     
mel 0.004s | enc 0.700s | prefill 0.983s | decode 3.818s | tokens 34 | tps 8.9
```cmd




# ???

<details>
<summary>Qwen 3 4B</summary>

```
```

</details>