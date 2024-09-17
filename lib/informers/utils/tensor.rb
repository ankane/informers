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

    def self.stack(tensors, dim = 0)
      tensors
    end

    def self.ones_like(tensor)
      if tensor[0].is_a?(Array)
        return tensor.map { |v| ones_like(v) }
      end
      tensor.map { |_| 1 }
    end

    def self.dims(tensor)
      dims = []
      while tensor.is_a?(Array)
        dims << tensor.size
        tensor = tensor[0]
      end
      dims
    end

    def self.interpolate(input, shape, mode = "bilinear", align_corners = false)
      out_height, out_width = shape

      # Input image dimensions
      in_channels = dims(input)[-3] || 1
      in_height = dims(input)[-2]
      in_width = dims(input)[-1]

      output = interpolate_data(
        input.flatten,
        [in_channels, in_height, in_width],
        [out_height, out_width],
        mode,
        align_corners
      )
      reshape(output, [in_channels, out_height, out_width])
    end

    def self.reshape(arr, dims)
      arr = arr.flatten
      dims[1..-1].reverse.each do |dim|
        arr = arr.each_slice(dim)
      end
      arr.to_a
    end
  end
end
