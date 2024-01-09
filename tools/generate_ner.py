from optimum.onnxruntime import ORTModelForTokenClassification
from pathlib import Path
import tempfile

dest = Path(tempfile.mkdtemp(), "ner.onnx")
model = ORTModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english", export=True)
model.save_pretrained(dest.parent)
dest.parent.joinpath("model.onnx").rename(dest)
print(dest)
