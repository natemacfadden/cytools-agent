# Harness image: the cytools-agent conda env + JupyterLab.
FROM condaforge/miniforge3

WORKDIR /app

# Build the conda env from the spec (CYTools and its deps come from there).
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

# The agent package.
COPY cytools_agent ./cytools_agent

# Serve JupyterLab, reachable from the host.
EXPOSE 8888
CMD ["conda", "run", "--no-capture-output", "-n", "cytools-agent", \
     "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root"]
