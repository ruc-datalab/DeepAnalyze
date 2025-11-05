# ============================================
# DeepAnalyze Docker Environment
# High-performance LLM inference with vLLM and CUDA 12.1
# ============================================

# Base image with NVIDIA CUDA support
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install vLLM
RUN pip3 install --upgrade pip && \
    pip3 install vllm

# Install common AI/ML tools
RUN pip3 install \
    torch \
    transformers \
    accelerate \
    fastapi \
    uvicorn

# Install data science and analysis tools
# Data processing and analysis
RUN pip3 install \
    pandasql \
    pyjanitor \
    feature-engine \
    missingno

# Data visualization
RUN pip3 install \
    plotly \
    bokeh \
    squarify \
    matplotlib-venn \
    wordcloud \
    folium \
    contextily

# Machine learning / Deep learning
RUN pip3 install \
    mlxtend \
    imbalanced-learn \
    optuna \
    shap \
    lime

# Time series and forecasting
RUN pip3 install \
    prophet \
    pmdarima \
    tslearn \
    lifelines

# Bayesian / Probabilistic modeling
RUN pip3 install \
    pymc \
    pystan \
    cmdstanpy \
    pyro-ppl

# Causal inference
RUN pip3 install \
    dowhy \
    econml \
    causalml \
    CausalInference \
    zepid

# Geospatial data science
RUN pip3 install \
    geopandas \
    shapely \
    rasterio \
    rasterstats \
    fiona \
    rtree \
    pysal \
    libpysal \
    esda \
    giddy \
    segregation \
    spopt \
    spreg \
    spglm \
    splot \
    spaghetti \
    tobler

# Expose port for API service
EXPOSE 8000

# Default command
CMD ["/bin/bash"]
