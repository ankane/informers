module Informers
  class FeatureExtractor
    def initialize(config)
      super()
      @config = config
    end
  end

  class ImageFeatureExtractor < FeatureExtractor
    def initialize(config)
      super(config)

      @image_mean = @config["image_mean"] || @config["mean"]
      @image_std = @config["image_std"] || @config["std"]

      @resample = @config["resample"] || 2 # 2 => bilinear
      @do_rescale = @config.fetch("do_rescale", true)
      @rescale_factor = @config["rescale_factor"] || (1 / 255.0)
      @do_normalize = @config["do_normalize"]

      @do_resize = @config["do_resize"]
      @do_thumbnail = @config["do_thumbnail"]
      @size = @config["size"]
      @size_divisibility = @config["size_divisibility"] || @config["size_divisor"]

      @do_center_crop = @config["do_center_crop"]
      @crop_size = @config["crop_size"]
      @do_convert_rgb = @config.fetch("do_convert_rgb", true)
      @do_crop_margin = @config["do_crop_margin"]

      @pad_size = @config["pad_size"]
      @do_pad = @config["do_pad"]

      if @do_pad && !@pad_size && @size && !@size["width"].nil? && !@size["height"].nil?
        # Should pad, but no pad size specified
        # We infer the pad size from the resize size
        @pad_size = @size
      end

      @do_flip_channel_order = @config["do_flip_channel_order"] || false
    end

    def rescale(pixel_data)
      pixel_data.length.times do |i|
        pixel_data[i] *= @rescale_factor
      end
    end

    def get_resize_output_image_size(image, size)
      if @config["keep_aspect_ratio"] && @config["ensure_multiple_of"]
        raise Todo
      end

      [size["width"], size["height"]]
    end

    def resize(image)
      new_width, new_height = get_resize_output_image_size(image, @size)
      image.resize(new_width, new_height, resample: @resample)
    end

    def preprocess(
      image,
      do_normalize: nil,
      do_pad: nil,
      do_convert_rgb: nil,
      do_convert_grayscale: nil,
      do_flip_channel_order: nil
    )
      if @do_crop_margin
        # NOTE: Specific to nougat processors. This is done before resizing,
        # and can be interpreted as a pre-preprocessing step.
        image = crop_margin(image)
      end

      src_width, src_height = image.size # original image size

      # Convert image to RGB if specified in config.
      # TODO

      # Resize all images
      if @do_resize
        image = resize(image)
      end

      # Resize the image using thumbnail method.
      if @do_thumbnail
        image = thumbnail(image, @size, @resample)
      end

      if @do_center_crop
        raise Todo
      end

      reshaped_input_size = [image.height, image.width]

      # NOTE: All pixel-level manipulation (i.e., modifying `pixelData`)
      # occurs with data in the hwc format (height, width, channels),
      # to emulate the behavior of the original Python code (w/ numpy).
      pixel_data = image.data
      img_dims = [image.height, image.width, image.channels]

      if @do_rescale
        rescale(pixel_data)
      end

      if !do_normalize.nil? ? do_normalize : @do_normalize
        image_mean = @image_mean
        if !@image_mean.is_a?(Array)
          image_mean = new Array(image.channels) { image_mean }
        end

        image_std = @image_std
        if !@image_std.is_a?(Array)
          image_std = new Array(image.channels) { image_std }
        end

        if image_mean.length != image.channels || image_std.length != image.channels
          raise Error, "When set to arrays, the length of `image_mean` (#{image_mean.length}) and `image_std` (#{image_std.length}) must match the number of channels in the image (#{image.channels})."
        end

        i = 0
        while i < pixel_data.length
          image.channels.times do |j|
            pixel_data[i + j] = (pixel_data[i + j] - image_mean[j]) / image_std[j];
          end
          i += image.channels
        end
      end

      # do padding after rescaling/normalizing
      if !do_pad.nil? ? do_pad : @do_pad
        raise Todo
      end

      if !do_flip_channel_order.nil? ? do_flip_channel_order : @do_flip_channel_order
        raise Todo
      end

      # convert to channel dimension format (hwc -> chw)
      h, w, c = img_dims
      pixel_values =
        c.times.map do |ci|
          h.times.map do |hi|
            w.times.map do |wi|
              index = (hi * w * c) + (wi * c) + ci
              pixel_data[index]
            end
          end
        end

      {
        original_size: [src_height, src_width],
        reshaped_input_size: reshaped_input_size,
        pixel_values: pixel_values
      }
    end

    def call(images, *args)
      if !images.is_a?(Array)
        images = [images]
      end

      image_data = images.map { |x| preprocess(x) }

      # Stack pixel values
      pixel_values = Utils.stack(image_data.map { |x| x[:pixel_values] }, 0);

      {
        pixel_values: pixel_values,

        # Original sizes of images
        original_sizes: image_data.map { |x| x[:original_size] },

        # Reshaped sizes of images, before padding or cropping
        reshaped_input_sizes: image_data.map { |x| x[:reshaped_input_size] }
      }
    end
  end

  class ViTFeatureExtractor < ImageFeatureExtractor
  end

  class Processor
    def initialize(feature_extractor)
      @feature_extractor = feature_extractor
    end

    def call(input, *args)
      @feature_extractor.(input, *args)
    end
  end

  class AutoProcessor
    FEATURE_EXTRACTOR_CLASS_MAPPING = {
      "ViTFeatureExtractor" => ViTFeatureExtractor
    }

    PROCESSOR_CLASS_MAPPING = {}

    def self.from_pretrained(
      pretrained_model_name_or_path,
      progress_callback: nil,
      config: nil,
      cache_dir: nil,
      local_files_only: false,
      revision: "main",
      **kwargs
    )
      preprocessor_config = config || Utils::Hub::get_model_json(pretrained_model_name_or_path, "preprocessor_config.json", true,
        progress_callback:,
        config:,
        cache_dir:,
        local_files_only:,
        revision:
      )

      # Determine feature extractor class
      # TODO: Ensure backwards compatibility with old configs
      key = preprocessor_config["feature_extractor_type"] || preprocessor_config["image_processor_type"]
      feature_extractor_class = FEATURE_EXTRACTOR_CLASS_MAPPING[key]

      if !feature_extractor_class
        if preprocessor_config["size"]
          # Assume ImageFeatureExtractor
          warn "Feature extractor type #{key.inspect} not found, assuming ImageFeatureExtractor due to size parameter in config."
          feature_extractor_class = ImageFeatureExtractor
        else
          raise Error, "Unknown Feature Extractor type: #{key}"
        end
      end

      # If no associated processor class, use default
      processor_class = PROCESSOR_CLASS_MAPPING[preprocessor_config["processor_class"]] || Processor

      # Instantiate processor and feature extractor
      feature_extractor = feature_extractor_class.new(preprocessor_config)
      processor_class.new(feature_extractor)
    end
  end
end
