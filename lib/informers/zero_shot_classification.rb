# Copyright 2018 The HuggingFace Inc. team.
# Copyright 2020 Andrew Kane.
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
  class ZeroShotClassification
    def initialize(model_path)
      tokenizer_path = File.expand_path("../../vendor/roberta_base.bin", __dir__)
      @tokenizer = BlingFire.load_model(tokenizer_path)
      @model = OnnxRuntime::Model.new(model_path)
    end
  end
end
