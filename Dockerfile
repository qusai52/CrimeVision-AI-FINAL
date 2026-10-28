FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/.cache/huggingface \
    MAX_SAMPLES=40 WHISPER_SIZE=base
RUN mkdir -p /home/user/app/data
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt gunicorn
# bake models into the image so the first analysis is not slow
RUN python -c "from huggingface_hub import snapshot_download as s; [s(m) for m in ['PekingU/rtdetr_r50vd','openai/clip-vit-base-patch32','Salesforce/blip-image-captioning-base','Systran/faster-whisper-base']]"
COPY --chown=user . .
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "1", "--threads", "4", "--timeout", "600", "app:app"]
