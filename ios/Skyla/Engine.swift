import Foundation

struct EngineError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

private final class TokenSink {
    let emit: (String, Bool) -> Void
    init(_ emit: @escaping (String, Bool) -> Void) { self.emit = emit }
}

private func takeError(_ pointer: UnsafeMutablePointer<CChar>?) -> EngineError {
    guard let pointer else { return EngineError(message: "unknown error") }
    defer { skyla_string_free(pointer) }
    return EngineError(message: String(cString: pointer))
}

final class Engine: @unchecked Sendable {
    private var handle: OpaquePointer?
    private(set) var modelName = ""
    private(set) var deviceName = ""

    func load(path: String, context: UInt32) throws {
        unload()
        var error: UnsafeMutablePointer<CChar>?
        guard let h = skyla_open(path, context, &error) else { throw takeError(error) }
        handle = h
        modelName = String(cString: skyla_model_name(h))
        if let d = skyla_device_name(h) {
            deviceName = String(cString: d)
            skyla_string_free(d)
        }
    }

    func unload() {
        if let h = handle { skyla_free(h) }
        handle = nil
    }

    func chat(_ message: String, system: String, temperature: Float, topP: Float, maxTokens: UInt32,
              onToken: @escaping (String, Bool) -> Void) throws -> SkylaStats {
        guard let h = handle else { throw EngineError(message: "no model loaded") }
        let sink = Unmanaged.passRetained(TokenSink(onToken))
        defer { sink.release() }
        var stats = SkylaStats()
        var error: UnsafeMutablePointer<CChar>?
        let callback: SkylaTokenCallback = { user, text, thinking in
            guard let user, let text else { return }
            Unmanaged<TokenSink>.fromOpaque(user).takeUnretainedValue().emit(String(cString: text), thinking)
        }
        let ok = skyla_chat(h, message, system, temperature, topP, maxTokens, callback,
                            sink.toOpaque(), &stats, &error)
        if !ok { throw takeError(error) }
        return stats
    }

    func cancel() {
        if let h = handle { skyla_cancel(h) }
    }

    func reset() {
        if let h = handle { skyla_reset(h) }
    }

    deinit { unload() }
}
