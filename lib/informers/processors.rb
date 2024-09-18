module Informers
  class FeatureExtractor
    attr_reader :config

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

    def thumbnail(image, size, resample = 2)
      input_height = image.height
      input_width = image.width

      output_height = size["height"]
      output_width = size["width"]

      # We always resize to the smallest of either the input or output size.
      height = [input_height, output_height].min
      width = [input_width, output_width].min

      if height == input_height && width == input_width
        return image
      end
      if input_height > input_width
        width = (input_width * height / input_height).floor
      elsif input_width > input_height
        height = (input_height * width / input_width).floor
      end
      image.resize(width, height, resample:)
    end

    def pad_image(
      pixel_data,
      img_dims,
      pad_size,
      mode: "constant",
      center: false,
      constant_values: 0
    )
      image_height, image_width, image_channels = img_dims

      if pad_size.is_a?(Numeric)
        padded_image_width = pad_size
        padded_image_height = pad_size
      else
        padded_image_width = pad_size[:width] || pad_size["width"]
        padded_image_height = pad_size[:height] || pad_size["height"]
      end

      # Only add padding if there is a difference in size
      if padded_image_width != image_width || padded_image_height != image_height
        padded_pixel_data = Array.new(padded_image_width * padded_image_height * image_channels)
        if constant_values.is_a?(Array)
          # Fill with constant values, cycling through the array
          padded_pixel_data.length.times do |i|
            padded_pixel_data[i] = constant_values[i % image_channels]
          end
        elsif constant_values != 0
          padded_pixel_data.fill(constant_values)
        end

        left, top =
          if center
            [((padded_image_width - image_width) / 2.0).floor, ((padded_image_height - image_height) / 2.0).floor]
          else
            [0, 0]
          end

        # Copy the original image into the padded image
        image_height.times do |i|
          a = (i + top) * padded_image_width
          b = i * image_width
          image_width.times do |j|
            c = (a + j + left) * image_channels
            d = (b + j) * image_channels
            image_channels.times do |k|
              padded_pixel_data[c + k] = pixel_data[d + k]
            end
          end
        end

        if mode == "symmetric"
          if center
            raise Error, "`center` padding is not supported when `mode` is set to `symmetric`."
          end
          h1 = image_height - 1
          w1 = image_width - 1
          padded_image_height.times do |i|
            a = i * padded_image_width
            b = Utils.calculate_reflect_offset(i, h1) * image_width

            padded_image_width.times do |j|
              next if i < image_height && j < image_width # Do not overwrite original image
              c = (a + j) * image_channels
              d = (b + Utils.calculate_reflect_offset(j, w1)) * image_channels

              # Copy channel-wise
              image_channels.times do |k|
                padded_pixel_data[c + k] = pixel_data[d + k]
              end
            end
          end
        end

        # Update pixel data and image dimensions
        pixel_data = padded_pixel_data
        img_dims = [padded_image_height, padded_image_width, image_channels]
      end
      [pixel_data, img_dims]
    end

    def rescale(pixel_data)
      pixel_data.length.times do |i|
        pixel_data[i] *= @rescale_factor
      end
    end

    def get_resize_output_image_size(image, size)
      src_width, src_height = image.size

      if @do_thumbnail
        # NOTE: custom logic for `Donut` models
        height = size["height"]
        width = size["width"]
        shortest_edge = [height, width].min
      elsif size.is_a?(Numeric)
        shortest_edge = size
        longest_edge = @config["max_size"] || shortest_edge
      elsif !size.nil?
        # Extract known properties from `size`
        shortest_edge = size["shortest_edge"]
        longest_edge = size["longest_edge"]
      end

      if !shortest_edge.nil? || !longest_edge.nil?
        # http://opensourcehacker.com/2011/12/01/calculate-aspect-ratio-conserving-resize-for-images-in-javascript/
        # Try resize so that shortest edge is `shortest_edge` (target)
        short_resize_factor =
          if shortest_edge.nil?
            1 # If `shortest_edge` is not set, don't upscale
          else
            [shortest_edge / src_width.to_f, shortest_edge / src_height.to_f].max
          end

        new_width = src_width * short_resize_factor
        new_height = src_height * short_resize_factor

        # The new width and height might be greater than `longest_edge`, so
        # we downscale again to ensure the largest dimension is `longest_edge`
        long_resize_factor =
          if longest_edge.nil?
            1 # If `longest_edge` is not set, don't downscale
          else
            [longest_edge / new_width.to_f, longest_edge / new_height.to_f].min
          end

        # To avoid certain floating point precision issues, we round to 2 decimal places
        final_width = (new_width * long_resize_factor).round(2).floor
        final_height = (new_height * long_resize_factor).round(2).floor

        if !@size_divisibility.nil?
          raise Todo
        end
        [final_width, final_height]
      elsif !size.nil? && !size["width"].nil? && !size["height"].nil?
        new_width = size["width"]
        new_height = size["height"]

        if @config["keep_aspect_ratio"] && @config["ensure_multiple_of"]
          raise Todo
        end

        [new_width, new_height]
      else
        raise Todo
      end
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
      if !do_convert_rgb.nil? ? do_convert_rgb : @do_convert_rgb
        image = image.rgb
      elsif do_convert_grayscale
        image = image.grayscale
      end

      # Resize all images
      if @do_resize
        image = resize(image)
      end

      # Resize the image using thumbnail method.
      if @do_thumbnail
        image = thumbnail(image, @size, @resample)
      end

      if @do_center_crop
        if @crop_size.is_a?(Integer)
          crop_width = @crop_size
          crop_height = @crop_size
        else
          crop_width = @crop_size["width"]
          crop_height = @crop_size["height"]
        end
        image = image.center_crop(crop_width, crop_height)
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
            pixel_data[i + j] = (pixel_data[i + j] - image_mean[j]) / image_std[j]
          end
          i += image.channels
        end
      end

      # do padding after rescaling/normalizing
      if !do_pad.nil? ? do_pad : @do_pad
        if @pad_size
          padded = pad_image(pixel_data, [image.height, image.width, image.channels], @pad_size)
          pixel_data, img_dims = padded # Update pixel data and image dimensions
        elsif @size_divisibility
          raise Todo
        end
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
      pixel_values = Utils.stack(image_data.map { |x| x[:pixel_values] }, 0)

      {
        pixel_values: pixel_values,

        # Original sizes of images
        original_sizes: image_data.map { |x| x[:original_size] },

        # Reshaped sizes of images, before padding or cropping
        reshaped_input_sizes: image_data.map { |x| x[:reshaped_input_size] }
      }
    end
  end

  class CLIPFeatureExtractor < ImageFeatureExtractor
  end

  class DPTFeatureExtractor < ImageFeatureExtractor
  end

  class ViTFeatureExtractor < ImageFeatureExtractor
  end

  class OwlViTFeatureExtractor < ImageFeatureExtractor
    def post_process_object_detection(*args)
      Utils.post_process_object_detection(*args)
    end
  end

  class Swin2SRImageProcessor < ImageFeatureExtractor
    def pad_image(pixel_data, img_dims, pad_size, **options)
      # NOTE: In this case, `padSize` represents the size of the sliding window for the local attention.
      # In other words, the image is padded so that its width and height are multiples of `padSize`.
      image_height, image_width, _image_channels = img_dims

      super(
        pixel_data,
        img_dims,
        {
          # NOTE: For Swin2SR models, the original python implementation adds padding even when the image's width/height is already
          # a multiple of `pad_size`. However, this is most likely a bug (PR: https://github.com/mv-lab/swin2sr/pull/19).
          # For this reason, we only add padding when the image's width/height is not a multiple of `pad_size`.
          width: image_width + (pad_size - image_width % pad_size) % pad_size,
          height: image_height + (pad_size - image_height % pad_size) % pad_size
        },
        mode: "symmetric",
        center: false,
        constant_values: -1,
        **options
      )
    end
  end

  class DonutFeatureExtractor < ImageFeatureExtractor
    def pad_image(pixel_data, img_dims, pad_size, **options)
      _image_height, _image_width, image_channels = img_dims

      image_mean = @image_mean
      if !image_mean.is_a?(Array)
        image_mean = new Array(image_channels, image_mean)
      end

      image_std = @image_std
      if !image_std.is_a?(Array)
        image_std = new Array(image_channels, image_std)
      end

      constant_values = image_mean.map.with_index { |x, i| -x / image_std[i] }

      super(
        pixel_data,
        img_dims,
        pad_size,
        center: true,
        # Since normalization is done after padding, we need to use certain constant values to ensure the same behaviour is observed.
        # For more information, see https://github.com/huggingface/transformers/blob/main/src/transformers/models/donut/image_processing_donut.py#L433-L451
        constant_values: constant_values,
        **options
      )
    end
  end

  class DetrFeatureExtractor < ImageFeatureExtractor
    def call(images)
      result = super(images)

      # TODO support differently-sized images, for now assume all images are the same size.
      # TODO support different mask sizes (not just 64x64)
      # Currently, just fill pixel mask with 1s
      mask_size = [result[:pixel_values].size, 64, 64]
      pixel_mask =
        mask_size[0].times.map do
          mask_size[1].times.map do
            mask_size[2].times.map do
              1
            end
          end
        end

      result.merge(pixel_mask: pixel_mask)
    end

    def post_process_object_detection(*args)
      Utils.post_process_object_detection(*args)
    end

    def remove_low_and_no_objects(class_logits, mask_logits, object_mask_threshold, num_labels)
      mask_probs_item = []
      pred_scores_item = []
      pred_labels_item = []

      class_logits.size.times do |j|
        cls = class_logits[j]
        mask = mask_logits[j]

        pred_label = Utils.max(cls)[1]
        if pred_label == num_labels
          # Is the background, so we ignore it
          next
        end

        scores = Utils.softmax(cls)
        pred_score = scores[pred_label]
        if pred_score > object_mask_threshold
          mask_probs_item << mask
          pred_scores_item << pred_score
          pred_labels_item << pred_label
        end
      end

      [mask_probs_item, pred_scores_item, pred_labels_item]
    end

    def check_segment_validity(
      mask_labels,
      mask_probs,
      k,
      mask_threshold = 0.5,
      overlap_mask_area_threshold = 0.8
    )
      # mask_k is a 1D array of indices, indicating where the mask is equal to k
      mask_k = []
      mask_k_area = 0
      original_area = 0

      mask_probs_k_data = mask_probs[k].flatten

      # Compute the area of all the stuff in query k
      mask_labels.length.times do |i|
        if mask_labels[i] == k
          mask_k << i
          mask_k_area += 1
        end

        if mask_probs_k_data[i] >= mask_threshold
          original_area += 1
        end
      end
      mask_exists = mask_k_area > 0 && original_area > 0

      # Eliminate disconnected tiny segments
      if mask_exists
        # Perform additional check
        area_ratio = mask_k_area / original_area
        mask_exists = area_ratio > overlap_mask_area_threshold
      end

      [mask_exists, mask_k]
    end

    def compute_segments(
      mask_probs,
      pred_scores,
      pred_labels,
      mask_threshold,
      overlap_mask_area_threshold,
      label_ids_to_fuse = nil,
      target_size = nil
    )
      height, width = target_size || Utils.dims(mask_probs[0])

      segmentation = Array.new(height * width)
      segments = []

      # 1. If target_size is not null, we need to resize the masks to the target size
      if !target_size.nil?
        # resize the masks to the target size
        mask_probs.length.times do |i|
          mask_probs[i] = Utils.interpolate(mask_probs[i], target_size, "bilinear", false)
        end
      end

      # 2. Weigh each mask by its prediction score
      # NOTE: `mask_probs` is updated in-place
      #
      # Temporary storage for the best label/scores for each pixel ([height, width]):
      mask_labels = Array.new(mask_probs[0].flatten.length)
      best_scores = Array.new(mask_probs[0].flatten.length, 0)

      mask_probs.length.times do |i|
        score = pred_scores[i]

        mask_probs_i_data = mask_probs[i].flatten
        mask_probs_i_dims = Utils.dims(mask_probs[i])

        mask_probs_i_data.length.times do |j|
          mask_probs_i_data[j] *= score
          if mask_probs_i_data[j] > best_scores[j]
            mask_labels[j] = i
            best_scores[j] = mask_probs_i_data[j]
          end
        end

        mask_probs[i] = Utils.reshape(mask_probs_i_data, mask_probs_i_dims)
      end

      current_segment_id = 0

      # stuff_memory_list = {}
      pred_labels.length.times do |k|
        pred_class = pred_labels[k]

        # TODO add `should_fuse`
        # should_fuse = label_ids_to_fuse.include?(pred_class)

        # Check if mask exists and large enough to be a segment
        mask_exists, mask_k = check_segment_validity(
          mask_labels,
          mask_probs,
          k,
          mask_threshold,
          overlap_mask_area_threshold
        )

        if !mask_exists
          # Nothing to see here
          next
        end

        current_segment_id += 1

        # Add current object segment to final segmentation map
        mask_k.each do |index|
          segmentation[index] = current_segment_id
        end

        segments << {
          id: current_segment_id,
          label_id: pred_class,
          score: pred_scores[k]
        }
      end

      segmentation = Utils.reshape(segmentation, [height, width])

      [segmentation, segments]
    end

    def post_process_panoptic_segmentation(
      outputs,
      threshold: 0.5,
      mask_threshold: 0.5,
      overlap_mask_area_threshold: 0.8,
      label_ids_to_fuse: nil,
      target_sizes: nil
    )
      if label_ids_to_fuse.nil?
        warn "`label_ids_to_fuse` unset. No instance will be fused."
        label_ids_to_fuse = Set.new
      end

      class_queries_logits = outputs[:logits] # [batch_size, num_queries, num_classes+1]
      masks_queries_logits = outputs[:pred_masks] # [batch_size, num_queries, height, width]

      mask_probs = Utils.sigmoid(masks_queries_logits) # [batch_size, num_queries, height, width]

      batch_size, _num_queries, num_labels = class_queries_logits.size, class_queries_logits[0].size, class_queries_logits[0][0].size
      num_labels -= 1 # Remove last class (background)

      if !target_sizes.nil? && target_sizes.length != batch_size
        raise Error, "Make sure that you pass in as many target sizes as the batch dimension of the logits"
      end

      to_return = []
      batch_size.times do |i|
        target_size = !target_sizes.nil? ? target_sizes[i] : nil

        class_logits = class_queries_logits[i]
        mask_logits = mask_probs[i]

        mask_probs_item, pred_scores_item, pred_labels_item = remove_low_and_no_objects(class_logits, mask_logits, threshold, num_labels)

        if pred_labels_item.length == 0
          raise Todo
        end

        # Get segmentation map and segment information of batch item
        segmentation, segments = compute_segments(
          mask_probs_item,
          pred_scores_item,
          pred_labels_item,
          mask_threshold,
          overlap_mask_area_threshold,
          label_ids_to_fuse,
          target_size
        )

        to_return << {
          segmentation: segmentation,
          segments_info: segments
        }
      end

      to_return
    end
  end

  module Utils
    def self.center_to_corners_format(v)
      centerX, centerY, width, height = v
      [
        centerX - width / 2.0,
        centerY - height / 2.0,
        centerX + width / 2.0,
        centerY + height / 2.0
      ]
    end

    def self.post_process_object_detection(outputs, threshold = 0.5, target_sizes = nil, is_zero_shot = false)
      out_logits = outputs[:logits]
      out_bbox = outputs[:pred_boxes]
      batch_size, num_boxes, num_classes = out_logits.size, out_logits[0].size, out_logits[0][0].size

      if !target_sizes.nil? && target_sizes.length != batch_size
        raise Error, "Make sure that you pass in as many target sizes as the batch dimension of the logits"
      end
      to_return = []
      batch_size.times do |i|
        target_size = !target_sizes.nil? ? target_sizes[i] : nil
        info = {
          boxes: [],
          classes: [],
          scores: []
        }
        logits = out_logits[i]
        bbox = out_bbox[i]

        num_boxes.times do |j|
          logit = logits[j]

          indices = []
          if is_zero_shot
            # Get indices of classes with high enough probability
            probs = Utils.sigmoid(logit)
            probs.length.times do |k|
              if probs[k] > threshold
                indices << k
              end
            end
          else
            # Get most probable class
            max_index = Utils.max(logit)[1]

            if max_index == num_classes - 1
              # This is the background class, skip it
              next
            end
            indices << max_index

            # Compute softmax over classes
            probs = Utils.softmax(logit)
          end

          indices.each do |index|
            box = bbox[j]

            # convert to [x0, y0, x1, y1] format
            box = center_to_corners_format(box)
            if !target_size.nil?
              box = box.map.with_index { |x, i| x * target_size[(i + 1) % 2] }
            end

            info[:boxes] << box
            info[:classes] << index
            info[:scores] << probs[index]
          end
        end
        to_return << info
      end
      to_return
    end
  end

  class WhisperFeatureExtractor < FeatureExtractor
    def initialize(config)
      super(config)

      raise Todo
    end

    def _extract_fbank_features(waveform)
      raise Todo
    end

    def call(audio)
      raise Todo
    end
  end

  class Wav2Vec2FeatureExtractor < FeatureExtractor
    def _zero_mean_unit_var_norm(input_values)
      sum = input_values.sum
      mean = sum / input_values.length.to_f
      variance = input_values.sum { |b| (b - mean) ** 2 } / input_values.length.to_f
      input_values.map { |x| (x - mean) / Math.sqrt(variance + 1e-7) }
    end

    def call(audio)
      # TODO
      # validate_audio_inputs(audio, 'Wav2Vec2FeatureExtractor')

      input_values = audio

      # zero-mean and unit-variance normalization
      if @config["do_normalize"]
        input_values = _zero_mean_unit_var_norm(input_values)
      end

      # TODO: allow user to pass in attention mask
      {
        input_values: [input_values],
        attention_mask: [Array.new(input_values.length, 1)]
      }
    end
  end

  class Processor
    attr_reader :feature_extractor

    def initialize(feature_extractor)
      @feature_extractor = feature_extractor
    end

    def call(input, *args)
      @feature_extractor.(input, *args)
    end
  end

  class AutoProcessor
    FEATURE_EXTRACTOR_CLASS_MAPPING = {
      "ViTFeatureExtractor" => ViTFeatureExtractor,
      "OwlViTFeatureExtractor" => OwlViTFeatureExtractor,
      "CLIPFeatureExtractor" => CLIPFeatureExtractor,
      "DPTFeatureExtractor" => DPTFeatureExtractor,
      "DetrFeatureExtractor" => DetrFeatureExtractor,
      "Swin2SRImageProcessor" => Swin2SRImageProcessor,
      "DonutFeatureExtractor" => DonutFeatureExtractor,
      "WhisperFeatureExtractor" => WhisperFeatureExtractor,
      "Wav2Vec2FeatureExtractor" => Wav2Vec2FeatureExtractor
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
