mkdir -p ./models
wget -P ./models https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
chmod +x ./models/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
./models/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile --server --nobrowser
