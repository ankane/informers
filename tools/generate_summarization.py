from pathlib import Path
import tempfile
from transformers.convert_graph_to_onnx import convert, quantize

dest = Path(tempfile.mkdtemp(), "summarization.onnx")
convert(
  pipeline_name="summarization",
  model="sshleifer/distilbart-cnn-12-6",
  output=dest,
  framework="pt",
  opset=11
)
quantize(dest)
