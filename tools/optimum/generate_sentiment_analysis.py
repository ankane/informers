from optimum.onnxruntime import ORTModelForSequenceClassification
from pathlib import Path
import tempfile

dest = Path(tempfile.mkdtemp(), "sentiment-analysis.onnx")
model = ORTModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", from_transformers=True)
model.save_pretrained(dest.parent, file_name=dest.name)
print(dest)
