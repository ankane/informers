from optimum.onnxruntime import ORTModelForMaskedLM
from pathlib import Path
import tempfile

dest = Path(tempfile.mkdtemp(), "fill-mask.onnx")
model = ORTModelForMaskedLM.from_pretrained("distilroberta-base", export=True)
model.save_pretrained(dest.parent)
dest.parent.joinpath("model.onnx").rename(dest)
print(dest)
