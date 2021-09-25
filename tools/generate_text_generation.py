from pathlib import Path
import tempfile
from transformers.convert_graph_to_onnx import convert, quantize

# requires:
# transformers==4.0.0
# torch==1.7.1

dest = Path(tempfile.mkdtemp(), "text-generation.onnx")
convert(
  pipeline_name="text-generation",
  model="gpt2",
  output=dest,
  framework="pt",
  opset=11
)
print(dest)
