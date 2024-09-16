module Informers
  module Utils
    class RawImage
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
        RawImage.new(@image.thumbnail_image(width, height: height, size: :force))
      end

      def rgb
        if @channels == 3
          return self
        end

        raise Todo
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
