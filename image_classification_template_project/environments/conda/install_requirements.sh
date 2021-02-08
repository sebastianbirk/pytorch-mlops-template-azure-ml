conda env create -f environments/conda/environment.yml --force
conda activate pytorch-aml-env
python -m ipykernel install --user --name=pytorch-aml-env
