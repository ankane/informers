from optimum.onnxruntime import ORTModelForFeatureExtraction
from pathlib import Path
import tempfile

dest = Path(tempfile.mkdtemp(), "feature-extraction.onnx")
model = ORTModelForFeatureExtraction.from_pretrained("distilbert-base-cased", from_transformers=True)
model.save_pretrained(dest.parent, file_name=dest.name)
print(dest)
