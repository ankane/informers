module Informers
  module Utils
    def self.dispatch_callback(progress_callback, data)
      progress_callback.(data) if progress_callback
    end

    def self.calculate_reflect_offset(i, w)
      ((i + w) % (2 * w) - w).abs
    end
  end
end
