# Copyright 2018 The HuggingFace Inc. team.
# Copyright 2021 Andrew Kane.
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
  class TextGeneration
    def initialize(model_path)
      encoder_path = File.expand_path("../../vendor/gpt2.bin", __dir__)
      @encoder = BlingFire.load_model(encoder_path, prefix: false)

      decoder_path = File.expand_path("../../vendor/gpt2.i2w", __dir__)
      @decoder = BlingFire.load_model(decoder_path)

      @model = OnnxRuntime::Model.new(model_path)
    end

    def predict(text, max_length: 50)
      tokens = @encoder.text_to_ids(text)

      input = {
        input_ids: [tokens]
      }
      if @model.inputs.any? { |i| i[:name] == "attention_mask" }
        input[:attention_mask] = [[1] * tokens.size]
      end

      output_name =
        if @model.outputs.any? { |o| o[:name] == "output_0" }
          "output_0"
        else
          "logits"
        end

      (max_length - tokens.size).times do |i|
        output = @model.predict(input, output_type: :numo, output_names: [output_name])
        # passed to input_ids
        tokens << output[output_name][0, true, true][-1, true].max_index
      end

      @decoder.ids_to_text(tokens)
    end
  end
end
