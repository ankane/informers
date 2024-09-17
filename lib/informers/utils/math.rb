module Informers
  module Utils
    def self.interpolate_data(input, in_shape, out_shape, mode = "bilinear", align_corners = false)
      in_channels, in_height, in_width = in_shape
      out_height, out_width = out_shape

      # TODO use mode and align_corners

      # Output image dimensions
      x_scale = out_width / in_width.to_f
      y_scale = out_height / in_height.to_f

      # Output image
      out_img = Array.new(out_height * out_width * in_channels)

      # Pre-calculate strides
      in_stride = in_height * in_width;
      out_stride = out_height * out_width;

      out_height.times do |i|
        out_width.times do |j|
          # Calculate output offset
          out_offset = i * out_width + j

          # Calculate input pixel coordinates
          x = (j + 0.5) / x_scale - 0.5
          y = (i + 0.5) / y_scale - 0.5

          # Calculate the four nearest input pixels
          # We also check if the input pixel coordinates are within the image bounds
          x1 = x.floor
          y1 = y.floor
          x2 = [x1 + 1, in_width - 1].min
          y2 = [y1 + 1, in_height - 1].min

          x1 = [x1, 0].max
          y1 = [y1, 0].max

          # Calculate the fractional distances between the input pixel and the four nearest pixels
          s = x - x1
          t = y - y1

          # Perform bilinear interpolation
          w1 = (1 - s) * (1 - t)
          w2 = s * (1 - t)
          w3 = (1 - s) * t
          w4 = s * t

          # Calculate the four nearest input pixel indices
          y_stride = y1 * in_width
          x_stride = y2 * in_width
          idx1 = y_stride + x1
          idx2 = y_stride + x2
          idx3 = x_stride + x1
          idx4 = x_stride + x2

          in_channels.times do |k|
            # Calculate channel offset
            c_offset = k * in_stride

            out_img[k * out_stride + out_offset] =
              w1 * input[c_offset + idx1] +
              w2 * input[c_offset + idx2] +
              w3 * input[c_offset + idx3] +
              w4 * input[c_offset + idx4]
          end
        end
      end

      out_img
    end

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
      if arr[0].is_a?(Array)
        return arr.map { |a| sigmoid(a) }
      end
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
