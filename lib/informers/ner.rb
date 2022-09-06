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
  class NER
    def initialize(model_path)
      tokenizer_path = File.expand_path("../../vendor/bert_base_cased_tok.bin", __dir__)
      @tokenizer = BlingFire.load_model(tokenizer_path)
      @model = OnnxRuntime::Model.new(model_path)
    end

    def predict(texts)
      singular = !texts.is_a?(Array)
      texts = [texts] if singular

      result = []
      texts.each do |text|
        # tokenize
        tokens, start_offsets, end_offsets = @tokenizer.text_to_ids_with_offsets(text, nil, 100) # unk token
        tokens.unshift(101) # cls token
        tokens << 102 # sep token

        # infer
        input = {
          input_ids: [tokens],
          attention_mask: [[1] * tokens.size],
          token_type_ids: [[0] * tokens.size]
        }
        res = @model.predict(input)

        # transform
        output = res["output_0"] || res["logits"]
        score =
          output[0].map do |e|
            values = e.map { |v| Math.exp(v) }
            sum = values.sum
            values.map { |v| v / sum }
          end

        labels_idx = score.map { |s| s.each_with_index.max[1] }
        labels = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

        entities = []
        filtered_label_idx = labels_idx.map.with_index.reject { |v, i| v == 0 }
        filtered_label_idx.each do |label_idx, idx|
          entities << {
            score: score[idx][label_idx],
            entity: labels[label_idx],
            index: idx
          }
        end

        result << group_entities(entities, text, start_offsets, end_offsets)
      end

      singular ? result.first : result
    end

    private

    def group_entities(entities, text, start_offsets, end_offsets)
      last_entity = {}
      groups = []
      entities.each do |entity|
        if entity[:index] - 1 == last_entity[:index] && entity[:entity] == last_entity[:entity]
          groups.last << entity
        else
          groups << [entity]
        end
        last_entity = entity
      end

      entity_map = {
        "I-PER" => "person",
        "I-ORG" => "org",
        "I-LOC" => "location",
        "I-MIS" => "misc"
      }

      groups.map do |group|
        start_offset = start_offsets[group.first[:index] - 1]
        end_offset = end_offsets[group.last[:index] - 1]

        {
          text: text[start_offset...end_offset],
          tag: entity_map[group.first[:entity]],
          score: group.map { |v| v[:score] }.sum / group.size,
          start: start_offset,
          end: end_offset
        }
      end
    end
  end
end
