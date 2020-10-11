from pathlib import Path
import tempfile
from transformers.convert_graph_to_onnx import convert, quantize

dest = Path(tempfile.mkdtemp(), "zero-shot-classification.onnx")
convert(
  pipeline_name="zero-shot-classification",
  model="facebook/bart-large-mnli",
  output=dest,
  framework="pt",
  opset=11
)
quantize(dest)
