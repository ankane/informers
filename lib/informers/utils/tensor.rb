module Informers
  module Utils
    def self.mean_pooling(last_hidden_state, attention_mask)
      last_hidden_state.zip(attention_mask).map do |state, mask|
        state[0].size.times.map do |k|
          sum = 0.0
          count = 0

          state.zip(mask) do |s, m|
            count += m
            sum += s[k] * m
          end

          sum / count
        end
      end
    end

    def self.normalize(result)
      result.map do |row|
        norm = Math.sqrt(row.sum { |v| v * v })
        row.map { |v| v / norm }
      end
    end
  end
end
