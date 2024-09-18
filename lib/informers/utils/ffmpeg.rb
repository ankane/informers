# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

module Informers
  module Utils
    # from the Transformers Python library
    def self.ffmpeg_read(data, sampling_rate)
      ar = "#{sampling_rate}"
      ac = "1"
      format_for_conversion = "f32le"
      ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1"
      ]

      stdout, status = Open3.capture2(*ffmpeg_command, stdin_data: data)
      if !status.success?
        raise Error, "ffmpeg was not found but is required to load audio files from filename"
      end
      stdout.unpack("f*")
    end
  end
end
