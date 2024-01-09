from optimum.onnxruntime import ORTModelForFeatureExtraction
from pathlib import Path
import tempfile

dest = Path(tempfile.mkdtemp(), "feature-extraction.onnx")
model = ORTModelForFeatureExtraction.from_pretrained("distilbert-base-cased", export=True)
model.save_pretrained(dest.parent)
dest.parent.joinpath("model.onnx").rename(dest)
print(dest)
