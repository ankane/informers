from pathlib import Path
import tempfile
from transformers.convert_graph_to_onnx import convert, quantize

dest = Path(tempfile.mkdtemp(), "sentiment-analysis.onnx")
convert(
  pipeline_name="sentiment-analysis",
  model="distilbert-base-uncased-finetuned-sst-2-english",
  output=dest,
  framework="pt",
  opset=11
)
quantize(dest)
