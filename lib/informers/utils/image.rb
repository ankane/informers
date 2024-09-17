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

        case resample_method
        when "bilinear", "bicubic"
          img =
            @image.affine(
              [width / @width.to_f, 0, 0, height / @height.to_f],
              interpolate: Vips::Interpolate.new(resample_method.to_sym)
            )
        else
          raise Todo
        end

        RawImage.new(img)
      end

      def center_crop(crop_width, crop_height)
        # If the image is already the desired size, return it
        if @width == crop_width && @height == crop_height
          return self
        end

        # Determine bounds of the image in the new canvas
        width_offset = (@width - crop_width) / 2.0
        height_offset = (@height - crop_height) / 2.0

        if width_offset >= 0 && height_offset >= 0
          # Cropped image lies entirely within the original image
          img = @image.crop(
            width_offset.floor,
            height_offset.floor,
            crop_width,
            crop_height
          )
        elsif width_offset <= 0 && height_offset <= 0
          raise Todo
        else
          raise Todo
        end

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

      def self.from_array(input)
        c, h, w = Utils.dims(input)
        pixel_data = Array.new(w * h * c)

        input.each_with_index do |cv, ci|
          cv.each_with_index do |hv, hi|
            hv.each_with_index do |v, wi|
              pixel_data[(hi * w * c) + (wi * c) + ci] = v
            end
          end
        end

        RawImage.new(Vips::Image.new_from_memory_copy(pixel_data.pack("C*"), w, h, c, :uchar))
      end
    end
  end
end
