# Exporting Models

```sh
git clone https://github.com/ankane/informers.git
cd informers/tools
pip3 install -r requirements.txt

# named-entity recognition
python3 generate_ner.py

# feature extraction
python3 generate_feature_extraction.py

# fill mask
python3 generate_fill_mask.py
```

The program will print the location of the model.

For text generation, use [this script](https://github.com/ankane/informers/blob/v0.2.0/tools/generate_text_generation.py).
