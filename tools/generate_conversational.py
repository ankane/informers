from pathlib import Path
import tempfile
from transformers.convert_graph_to_onnx import convert, quantize

dest = Path(tempfile.mkdtemp(), "conversational.onnx")
convert(
  pipeline_name="conversational",
  model="microsoft/DialoGPT-medium",
  output=dest,
  framework="pt",
  opset=11
)
quantize(dest)
