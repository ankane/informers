module Informers
  module Utils
    def self.softmax(arr)
      # Compute the maximum value in the array
      max_val = arr.max

      #  Compute the exponentials of the array values
      exps = arr.map { |x| Math.exp(x - max_val) }

      # Compute the sum of the exponentials
      sum_exps = exps.sum

      # Compute the softmax values
      softmax_arr = exps.map { |x| x / sum_exps }

      softmax_arr
    end

    def self.sigmoid(arr)
      arr.map { |v| 1 / (1 + Math.exp(-v)) }
    end

    def self.get_top_items(items, top_k = 0)
      # if top == 0, return all

      items = items
        .map.with_index { |x, i| [i, x] } # Get indices ([index, score])
        .sort_by { |v| -v[1] }            # Sort by log probabilities

      if !top_k.nil? && top_k > 0
        items = items.slice(0, top_k)     # Get top k items
      end

      items
    end

    def self.max(arr)
      if arr.length == 0
        raise Error, "Array must not be empty"
      end
      arr.map.with_index.max_by { |v, _| v }
    end
  end
end
