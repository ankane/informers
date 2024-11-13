module Informers
  module Backends
    module Onnx
      def self.device_to_execution_providers(device)
        case device&.to_s
        when "cpu", nil
          []
        when "cuda"
          ["CUDAExecutionProvider"]
        when "coreml"
          ["CoreMLExecutionProvider"]
        else
          raise ArgumentError, "Unsupported device: #{device.inspect}"
        end
      end
    end
  end
end
