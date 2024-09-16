module Informers
  module Utils
    class RawImage
      RESAMPLING_MAPPING = {
        0 => "nearest",
        1 => "lanczos",
        2 => "bilinear",
        3 => "bicubic",
        4 => "box",
        5 => "hamming",
      }

      attr_reader :width, :height, :channels

      def initialize(image)
        @image = image
        @width = image.width
        @height = image.height
        @channels = image.bands
      end

      def data
        @image.write_to_memory.unpack("C*")
      end

      def size
        [@width, @height]
      end

      def resize(width, height, resample: 2)
        resample_method = RESAMPLING_MAPPING[resample] || resample

        if resample_method != "bilinear"
          raise Todo
        end

        img =
          @image.affine(
            [width / @width.to_f, 0, 0, height / @height.to_f],
            interpolate: Vips::Interpolate.new(:bilinear)
          )
        RawImage.new(img)
      end

      def rgb
        if @channels == 3
          return self
        end

        raise Todo
      end

      def save(path)
        @image.write_to_file(path)
      end

      def self.read(input)
        if input.is_a?(RawImage)
          input
        elsif input.is_a?(URI)
          require "open-uri"

          RawImage.new(Vips::Image.new_from_buffer(input.read, ""))
        elsif input.is_a?(String)
          RawImage.new(Vips::Image.new_from_file(input))
        else
          raise ArgumentError, "Unsupported input type: #{input.class.name}"
        end
      end
    end
  end
end
