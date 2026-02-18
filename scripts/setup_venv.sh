cd /workspace/pulpy

uv venv
source .venv/bin/activate

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

cd /workspace/pulpy/code/Umamba2

uv pip install -e .
uv pip install wheel
uv pip install causal-conv1d==1.5.2 mamba-ssm==2.2.5 --no-build-isolation

uv pip install -U "transformers==4.47.1"

cd /workspace/pulpy

echo "READY."