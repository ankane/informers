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
  class QuestionAnswering
    def initialize(model_path)
      tokenizer_path = File.expand_path("../../vendor/bert_base_cased_tok.bin", __dir__)
      @tokenizer = BlingFire.load_model(tokenizer_path)
      @model = OnnxRuntime::Model.new(model_path)
    end

    def predict(questions)
      singular = !questions.is_a?(Array)
      questions = [questions] if singular

      topk = 1
      max_answer_len = 15

      sep_pos = []
      cls_pos = []
      context_offsets = []

      # tokenize
      input_ids =
        questions.map do |question|
          tokens = @tokenizer.text_to_ids(question[:question], nil, 100) # unk token
          sep_pos << tokens.size
          tokens << 102 # sep token
          context_tokens, offsets = @tokenizer.text_to_ids_with_offsets(question[:context], nil, 100) # unk token
          tokens.concat(context_tokens)
          context_offsets << offsets
          cls_pos << tokens.size
          tokens.unshift(101) # cls token
          tokens << 102 # sep token
          tokens
        end

      max_tokens = 384
      raise "Large text not supported yet" if input_ids.map(&:size).max > max_tokens

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

      start = output["output_0"] || output["start_logits"]
      stop = output["output_1"] || output["end_logits"]

      # transform
      answers = []
      start.zip(stop).each_with_index do |(start_, end_), i|
        start_ = Numo::DFloat.cast(start_)
        end_ = Numo::DFloat.cast(end_)

        # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
        feature_p_mask = Numo::Int64.new(max_tokens).fill(0)
        feature_p_mask[1..sep_pos[i] + 1] = 1
        feature_p_mask[cls_pos[i][1]] = 1
        feature_attention_mask = Numo::Int64.cast(attention_mask[i])
        undesired_tokens = (feature_p_mask - 1).abs & feature_attention_mask

        # Generate mask
        undesired_tokens_mask = undesired_tokens.eq(0)

        # Make sure non-context indexes in the tensor cannot contribute to the softmax
        start_[undesired_tokens_mask] = -10000
        end_[undesired_tokens_mask] = -10000

        # Normalize logits and spans to retrieve the answer
        start_ = Numo::DFloat::Math.exp(start_ - Numo::DFloat::Math.log(Numo::DFloat::Math.exp(start_).sum(axis: -1)))
        end_ = Numo::DFloat::Math.exp(end_ - Numo::DFloat::Math.log(Numo::DFloat::Math.exp(end_).sum(axis: -1)))

        # Mask CLS
        start_[0] = end_[0] = 0.0

        starts, ends, scores = decode(start_, end_, topk, max_answer_len)

        # char_to_word
        doc_tokens, char_to_word_offset = send(:doc_tokens, questions[i][:context])
        char_to_word = Numo::Int64.cast(char_to_word_offset)

        # token_to_orig_map
        token_to_orig_map = {}
        map_pos = sep_pos[i] + 2
        context_offsets[i].each do |offset|
          token_to_orig_map[map_pos] = char_to_word_offset[offset]
          map_pos += 1
        end

        # Convert the answer (tokens) back to the original text
        starts.to_a.zip(ends.to_a, scores) do |s, e, score|
          answers << {
            answer: doc_tokens[token_to_orig_map[s]..token_to_orig_map[e]].join(" "),
            score: score,
            start: (char_to_word.eq(token_to_orig_map[s])).where[0],
            end: (char_to_word.eq(token_to_orig_map[e])).where[-1]
          }
        end
      end

      singular ? answers.first : answers
    end

    private

    def decode(start, stop, topk, max_answer_len)
      # Ensure we have batch axis
      if start.ndim == 1
        start = start.expand_dims(0)
      end

      if stop.ndim == 1
        stop = stop.expand_dims(0)
      end

      # Compute the score of each tuple(start, end) to be the real answer
      outer = start.expand_dims(-1).dot(stop.expand_dims(1))

      # Remove candidate with end < start and end - start > max_answer_len
      candidates = outer.triu.tril(max_answer_len - 1)

      # Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
      scores_flat = candidates.flatten
      if topk == 1
        idx_sort = [scores_flat.argmax]
      else
        raise "Not implemented yet"
      end

      start, stop = unravel_index(idx_sort, candidates.shape)[1..-1]
      [start, stop, candidates[0, start, stop]]
    end

    def unravel_index(indices, shape)
      indices = Numo::NArray.cast(indices)
      result = []
      factor = 1
      shape.size.times do |i|
        result.unshift(indices / factor % shape[-1 - i])
        factor *= shape[-1 - i]
      end
      result
    end

    def doc_tokens(text)
      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = true

      text.each_char do |c|
        if whitespace?(c)
          prev_is_whitespace = true
        else
          if prev_is_whitespace
            doc_tokens << c
          else
            doc_tokens[-1] += c
          end
          prev_is_whitespace = false
        end
        char_to_word_offset << (doc_tokens.size - 1)
      end
      # ensure end is correct when answer includes last token
      char_to_word_offset << (doc_tokens.size - 1)

      [doc_tokens, char_to_word_offset]
    end

    def whitespace?(c)
      c == " " || c == "\t" || c == "\r" || c == "\n" || c.ord == 0x202F
    end
  end
end
