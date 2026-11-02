import Foundation
import Metal
import SwiftUI

enum Phase: Equatable {
    case idle
    case downloading(Double)
    case loading
    case ready
    case generating
}

@MainActor
final class ChatModel: ObservableObject {
    @Published var messages: [MessageModel] = []
    @Published var phase: Phase = .idle
    @Published var models: [ModelOption] = Models.all()
    @Published var current: ModelOption?
    @Published var status = "Pick a model to start"
    @Published var problem: String?
    @Published var loadSeconds: Double?

    nonisolated static let gpuName: String = MTLCreateSystemDefaultDevice()?.name ?? "Apple GPU"
    nonisolated static var chip: String {
        gpuName.replacingOccurrences(of: "Apple ", with: "").replacingOccurrences(of: " GPU", with: "")
    }
    @AppStorage("temperature") var temperature = 0.7
    @AppStorage("maxTokens") var maxTokens = 1024
    @AppStorage("system") var system = ""
    @AppStorage("lastModel") private var lastModel = ""

    private let engine = Engine()
    private let queue = DispatchQueue(label: "skyla.engine", qos: .userInitiated)
    private let downloader = Downloader()
    nonisolated static let context: UInt32 = 4096

    init() {
        if let last = models.first(where: { $0.id == lastModel && $0.isDownloaded }) {
            load(last)
        }
    }

    var busy: Bool {
        switch phase {
        case .downloading, .loading, .generating: true
        default: false
        }
    }

    func refresh() {
        models = Models.all()
    }

    func select(_ option: ModelOption) {
        if option.isDownloaded {
            load(option)
        } else {
            download(option)
        }
    }

    private func download(_ option: ModelOption) {
        phase = .downloading(0)
        status = "Downloading \(option.name)…"
        downloader.start(option, progress: { p in
            Task { @MainActor in self.phase = .downloading(p) }
        }, finish: { result in
            Task { @MainActor in
                self.refresh()
                switch result {
                case .success: self.load(option)
                case .failure(let e):
                    self.phase = .idle
                    self.status = "Download failed: \(e.localizedDescription)"
                    self.problem = self.status
                }
            }
        })
    }

    func load(_ option: ModelOption) {
        phase = .loading
        status = "Loading \(option.name)…"
        messages = []
        let engine = engine
        let path = option.file.path
        queue.async {
            let started = Date()
            let result = Result { try engine.load(path: path, context: Self.context) }
            let seconds = Date().timeIntervalSince(started)
            Task { @MainActor in
                switch result {
                case .success:
                    self.current = option
                    self.lastModel = option.id
                    self.phase = .ready
                    self.status = "\(engine.deviceName) · loaded in \(String(format: "%.1f", seconds)) s"
                    self.loadSeconds = seconds
                    self.problem = nil
                case .failure(let e):
                    self.current = nil
                    self.phase = .idle
                    self.status = e.localizedDescription
                    self.problem = e.localizedDescription
                }
            }
        }
    }

    func send(_ text: String) {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, phase == .ready else { return }
        let reply = MessageModel(user: false, streaming: true)
        messages.append(MessageModel(user: true, text: text))
        messages.append(reply)
        phase = .generating
        let pending = PendingText()
        let (engine, system) = (engine, system)
        let (temperature, maxTokens) = (Float(temperature), UInt32(maxTokens))
        Task { @MainActor in
            while true {
                let chunk = pending.take()
                reply.receive(chunk)
                if !reply.reveal(finished: chunk.finished) { break }
                try? await Task.sleep(for: .milliseconds(16))
            }
        }
        queue.async {
            let result = Result {
                try engine.chat(text, system: system, temperature: temperature, topP: 0.95,
                                maxTokens: maxTokens) { piece, thinking in
                    pending.append(piece, thinking: thinking)
                }
            }
            pending.finish()
            Task { @MainActor in
                switch result {
                case .success(let s):
                    reply.stats = String(
                        format: "%d tokens · %.1f tok/s · %d/%d context",
                        s.tokens, s.tokens_per_second, s.position, s.context)
                case .failure(let e):
                    reply.error = e.localizedDescription
                }
                self.phase = .ready
            }
        }
    }

    func stop() {
        engine.cancel()
    }

    func newChat() {
        guard phase == .ready else { return }
        let engine = engine
        queue.async { engine.reset() }
        messages = []
    }

    func delete(_ option: ModelOption) {
        guard option != current, !busy else { return }
        try? FileManager.default.removeItem(at: option.file)
        refresh()
    }

    func flush() {
        guard phase == .ready else { return }
        let engine = engine
        queue.async { engine.unload() }
        current = nil
        messages = []
        phase = .idle
        status = "Memory flushed · pick a model to load it again"
    }
}
