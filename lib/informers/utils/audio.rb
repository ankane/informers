module Informers
  module Utils
    def self.read_audio(input, sampling_rate)
      data =
        if input.is_a?(URI)
          require "open-uri"

          input.read
        elsif input.is_a?(String)
          File.binread(input)
        else
          raise ArgumentError, "Unsupported input type: #{input.class.name}"
        end

      ffmpeg_read(data, sampling_rate)
    end
  end
end
