cd /workspace/pulpy

uv venv
source .venv/bin/activate

pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

cd /workspace/pulpy/code/Umamba2

uv pip install -e .
uv pip install wheel
uv pip install causal-conv1d==1.5.2 mamba-ssm==2.2.5 --no-build-isolation

uv pip install -U "transformers==4.47.1"

cd /workspace/pulpy

echo "READY."