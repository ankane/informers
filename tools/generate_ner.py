from pathlib import Path
import tempfile
from transformers.convert_graph_to_onnx import convert, quantize

dest = Path(tempfile.mkdtemp(), "ner.onnx")
convert(
  pipeline_name="ner",
  model="dbmdz/bert-large-cased-finetuned-conll03-english",
  output=dest,
  framework="pt",
  opset=11
)
quantize(dest)
