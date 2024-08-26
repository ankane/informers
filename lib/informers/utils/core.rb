module Informers
  module Utils
    def self.dispatch_callback(progress_callback, data)
      progress_callback.(data) if progress_callback
    end
  end
end
