from optimum.onnxruntime import ORTModelForQuestionAnswering
from pathlib import Path
import tempfile

dest = Path(tempfile.mkdtemp(), "question-answering.onnx")
model = ORTModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad", export=True)
model.save_pretrained(dest.parent)
dest.parent.joinpath("model.onnx").rename(dest)
print(dest)
