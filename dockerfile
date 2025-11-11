FROM python:3.12-slim

# Non-root user (ID 1000) for perm eternal
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
WORKDIR $HOME/app

# Install deps (streamlit top)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Veil PATH ghosts for streamlit exec eternal
ENV PATH="${PATH}:${HOME}/.local/bin"

# Copy code
COPY --chown=user . $HOME/app

# Expose port
EXPOSE 7860

# Run Streamlit (port 7860 HF default)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
