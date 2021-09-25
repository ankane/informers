from pathlib import Path
import tempfile
from transformers.convert_graph_to_onnx import convert, quantize

dest = Path(tempfile.mkdtemp(), "fill-mask.onnx")
convert(
  pipeline_name="fill-mask",
  model="distilroberta-base",
  output=dest,
  framework="pt",
  opset=11
)
# quantize(dest)
