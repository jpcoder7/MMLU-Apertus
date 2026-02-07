# MMLU-Apertus
- Mit MMLU-Apertus kann man drei verschiedene Optimierungsmethoden austesten (Prompt Engineering, RAG (Retrieval-Augmented Generation) und RAG mit Query Reformulation (in der Theorie sollte es für eine effizientere Suche in der Datenbank sorgen)
- Alle notwendige Skripte befinden sich in diesem Repo 
- Mit dieser Anleitung kann man die Umgebung (idealerweise auf Runpod - https://www.runpod.io) einrichten 

# Setup-Anleitung
Dies sind die erforderlichen Schritte, um eine geeignete Umgebung für die Evaluation zu erstellen.

# 1. Umgebung einrichten
1. Cache-Verzeichnisse konfigurieren (wichtig für RunPod)
   export HF_HOME=/workspace/.cache/huggingface
   export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

2. Hauptabhängigkeiten installieren

   pip install transformers==4.56.2 accelerate safetensors

3. Evaluation-Framework installieren

   pip install "lm-eval[wandb]" datasets sentencepiece

4. RAG-Abhängigkeiten installieren

   pip install faiss-cpu sentence-transformers
   
# 2. Wikipedia-Index erstellen

1. Wikipedia herunterladen (ca. 20 GB)
   
   python download_wiki.py bzw. den Befehl direkt im Terminal eingeben

2. Daten aufbereiten
python prepare_wiki.py \
--input_jsonl /workspace/rag/wiki/enwiki_20231101_preprocessed.jsonl \
--out_jsonl /workspace/rag/wiki/enwiki_prepared.jsonl

4. FAISS-Index erstellen
python build_faiss.py \
--corpus_jsonl /workspace/rag/wiki/enwiki_prepared.jsonl \
--out_dir /workspace/rag/wiki \
--batch_size 128

#3. Tests durchführen

Baseline-Test (So startet man den normalen MMLU Test - Parameter können angepasst werden)
lm-eval \
--model hf \
--model_args pretrained=swiss-ai/Apertus-8B-2509,trust_remote_code=True \
--tasks mmlu \
--num_fewshot 0 \
--batch_size auto \
--output_path /workspace/outputs/baseline.json

Prompt Test (Parameter können angepasst werden)
python prompt.py \
--pretrained swiss-ai/Apertus-8B-2509 \
--tasks mmlu \
--num_fewshot 0 \
--limit 0 \
--prompt_preamble "You are taking a multiple-choice exam.." \ (man erstellt eine Textdatei (txt) und referenziert diese)
--output_path /workspace/outputs/prompt_default.json

RAG Test (Parameter können angepasst werden)
python rag.py \
--index_dir /workspace/rag/wiki \
--pretrained swiss-ai/Apertus-8B-2509 \
--tasks mmlu \
--num_fewshot 0 \
--top_k 3 \
--max_ctx_chars 2400 \
--output_path /workspace/outputs/rag_3_2400.json

Rewrite Test (Parameter können angepasst werden)
python rewrite.py \
--index_dir /workspace/rag/wiki \
--pretrained swiss-ai/Apertus-8B-2509 \
--tasks mmlu \
--num_fewshot 0 \
--top_k 3 \
--max_ctx_chars 2400 \
--rewrite_max_new_tokens 32 \
--output_path /workspace/outputs/qr_32.json
