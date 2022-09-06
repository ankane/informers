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
  class FillMask
    def initialize(model_path)
      encoder_path = File.expand_path("../../vendor/roberta.bin", __dir__)
      @encoder = BlingFire.load_model(encoder_path, prefix: false)

      decoder_path = File.expand_path("../../vendor/roberta.i2w", __dir__)
      @decoder = BlingFire.load_model(decoder_path)

      @model = OnnxRuntime::Model.new(model_path)
    end

    def predict(texts)
      singular = !texts.is_a?(Array)
      texts = [texts] if singular

      mask_token = 50264

      # tokenize
      input_ids =
        texts.map do |text|
          tokens = @encoder.text_to_ids(text, nil, 3) # unk token

          # add mask token
          mask_sequence = [28696, 43776, 15698]
          masks = []
          (tokens.size - 2).times do |i|
            masks << i if tokens[i..(i + 2)] == mask_sequence
          end
          masks.reverse.each do |mask|
            tokens = tokens[0...mask] + [mask_token] + tokens[(mask + 3)..-1]
          end

          tokens.unshift(0) # cls token
          tokens << 2 # sep token

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

      input = {
        input_ids: input_ids,
        attention_mask: attention_mask
      }

      masked_index = input_ids.map { |v| v.each_index.select { |i| v[i] == mask_token } }
      masked_index.each do |v|
        raise "No mask_token (<mask>) found on the input" if v.size < 1
        raise "More than one mask_token (<mask>) is not supported" if v.size > 1
      end

      res = @model.predict(input)
      outputs = res["output_0"] || res["logits"]
      batch_size = outputs.size

      results = []
      batch_size.times do |i|
        result = []

        logits = outputs[i][masked_index[i][0]]
        values = logits.map { |v| Math.exp(v) }
        sum = values.sum
        probs = values.map { |v| v / sum }
        res = probs.each_with_index.sort_by { |v| -v[0] }.first(5)

        res.each do |(v, p)|
          tokens = input[:input_ids][i].dup
          tokens[masked_index[i][0]] = p
          result << {
            sequence: @decoder.ids_to_text(tokens),
            score: v,
            token: p,
            # TODO figure out prefix space
            token_str: @decoder.ids_to_text([p], skip_special_tokens: false)
          }
        end

        results += [result]
      end

      singular ? results.first : results
    end
  end
end
