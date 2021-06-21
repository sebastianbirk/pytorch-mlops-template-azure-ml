FROM conda/miniconda3

COPY mlops_pipeline_environment.yml /setup/

# activate environment
ENV PATH /usr/local/envs/mlops_pipeline_env/bin:$PATH

RUN conda update -n base -c defaults conda && \
    conda install python=3.7.5 && \
    conda env create -f /setup/mlops_pipeline_environment.yml && \
    /bin/bash -c "source activate mlops_pipeline_env" && \
    az --version && \
    chmod -R 777 /usr/local/envs/mlops_pipeline_env/lib/python3.7