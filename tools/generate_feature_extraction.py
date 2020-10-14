from pathlib import Path
import tempfile
from transformers.convert_graph_to_onnx import convert, quantize

dest = Path(tempfile.mkdtemp(), "feature-extraction.onnx")
convert(
  pipeline_name="feature-extraction",
  model="distilbert-base-cased",
  output=dest,
  framework="pt",
  opset=11
)
quantize(dest)
