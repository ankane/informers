# Copyright 2018 The HuggingFace Inc. team.
# Copyright 2020 Andrew Kane.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

module Informers
  class FeatureExtraction
    def initialize(model_path)
      tokenizer_path = File.expand_path("../../vendor/bert_base_cased_tok.bin", __dir__)
      @tokenizer = BlingFire.load_model(tokenizer_path)
      @model = OnnxRuntime::Model.new(model_path)
    end

    def predict(texts)
      singular = !texts.is_a?(Array)
      texts = [texts] if singular

      # tokenize
      input_ids =
        texts.map do |text|
          tokens = @tokenizer.text_to_ids(text, nil, 100) # unk token
          tokens.unshift(101) # cls token
          tokens << 102 # sep token
          tokens
        end

      max_tokens = input_ids.map(&:size).max
      attention_mask = []
      input_ids.each do |ids|
        zeros = [0] * (max_tokens - ids.size)

        mask = ([1] * ids.size) + zeros
        attention_mask << mask

        ids.concat(zeros)
      end

      # infer
      input = {
        input_ids: input_ids,
        attention_mask: attention_mask
      }
      output = @model.predict(input)
      scores = output["output_0"] || output["last_hidden_state"]

      singular ? scores.first : scores
    end
  end
end
