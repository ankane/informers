from optimum.onnxruntime import ORTModelForCausalLM
from pathlib import Path
import tempfile

# TODO get working in Ruby

dest = Path(tempfile.mkdtemp(), "text-generation.onnx")
model = ORTModelForCausalLM.from_pretrained("gpt2", export=True)
model.save_pretrained(dest.parent)
dest.parent.joinpath("model.onnx").rename(dest)
print(dest)
