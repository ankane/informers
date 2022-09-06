from optimum.onnxruntime import ORTModelForCausalLM
from pathlib import Path
import tempfile

# TODO get working

dest = Path(tempfile.mkdtemp(), "text-generation.onnx")
model = ORTModelForCausalLM.from_pretrained("gpt2", from_transformers=True)
model.save_pretrained(dest.parent, file_name=dest.name)
print(dest)
