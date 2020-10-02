from pathlib import Path
import tempfile
from transformers.convert_graph_to_onnx import convert, quantize

dest = Path(tempfile.mkdtemp(), "question-answering.onnx")
convert(
  pipeline_name="question-answering",
  model="distilbert-base-cased-distilled-squad",
  output=dest,
  framework="pt",
  opset=11
)
quantize(dest)
