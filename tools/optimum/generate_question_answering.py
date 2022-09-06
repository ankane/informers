from optimum.onnxruntime import ORTModelForQuestionAnswering
from pathlib import Path
import tempfile

dest = Path(tempfile.mkdtemp(), "question-answering.onnx")
model = ORTModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad", from_transformers=True)
model.save_pretrained(dest.parent, file_name=dest.name)
print(dest)
