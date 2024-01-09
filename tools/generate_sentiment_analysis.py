from optimum.onnxruntime import ORTModelForSequenceClassification
from pathlib import Path
import tempfile

dest = Path(tempfile.mkdtemp(), "sentiment-analysis.onnx")
model = ORTModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", export=True)
model.save_pretrained(dest.parent)
dest.parent.joinpath("model.onnx").rename(dest)
print(dest)
