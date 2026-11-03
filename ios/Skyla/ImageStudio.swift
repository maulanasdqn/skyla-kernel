import CoreML
import Foundation
import StableDiffusion
import SwiftUI
import UIKit

struct GeneratedImage: Identifiable, Hashable {
    let url: URL
    var id: URL { url }
}

private final class Generator: @unchecked Sendable {
    private var pipeline: (any StableDiffusionPipelineProtocol)?
    private(set) var loadedID: String?
    var cancelled = false

    func load(_ model: ImageModel) throws {
        if loadedID == model.id { return }
        pipeline = nil
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let p: any StableDiffusionPipelineProtocol = model.isXL
            ? try StableDiffusionXLPipeline(resourcesAt: model.directory, configuration: config, reduceMemory: true)
            : try StableDiffusionPipeline(resourcesAt: model.directory, controlNet: [], configuration: config,
                                          disableSafety: true, reduceMemory: true)
        try p.loadResources()
        pipeline = p
        loadedID = model.id
    }

    func unload() {
        pipeline = nil
        loadedID = nil
    }

    func generate(_ model: ImageModel, prompt: String, negative: String, steps: Int, guidance: Float, seed: UInt32,
                  progress: @escaping (Int, Int) -> Void) throws -> CGImage? {
        guard let pipeline else { return nil }
        cancelled = false
        var config = PipelineConfiguration(prompt: prompt)
        config.negativePrompt = negative
        config.stepCount = steps
        config.guidanceScale = guidance
        config.seed = seed
        config.disableSafety = true
        config.schedulerType = .dpmSolverMultistepScheduler
        config.schedulerTimestepSpacing = .karras
        let scale: Float32 = model.isXL ? 0.13025 : 0.18215
        config.encoderScaleFactor = scale
        config.decoderScaleFactor = scale
        let images = try pipeline.generateImages(configuration: config) { p in
            progress(p.step + 1, p.stepCount)
            return !self.cancelled
        }
        return images.first ?? nil
    }
}

@MainActor
final class ImageStudio: ObservableObject {
    enum State: Equatable {
        case idle
        case downloading(Double)
        case loading
        case remote
        case generating(Int, Int)
    }

    @Published var state: State = .idle
    @Published var model = ImageModels.catalog[0]
    @Published var gallery: [GeneratedImage] = []
    @Published var latest: GeneratedImage?
    @Published var problem: String?
    @Published var downloaded: Set<String> = []
    @Published var falKey: String = Keychain.get() {
        didSet { Keychain.set(falKey.trimmingCharacters(in: .whitespacesAndNewlines)) }
    }
    @AppStorage("imageSteps") var steps = 20
    @AppStorage("imageGuidance") var guidance = 7.0
    @AppStorage("imageNegative") var negative = "blurry, low quality, deformed, watermark, text"

    var freeMemory: (() -> Void)?
    private let generator = Generator()
    private let queue = DispatchQueue(label: "skyla.images", qos: .userInitiated)
    static let outputs = Models.directory.appendingPathComponent("Images", isDirectory: true)

    init() {
        refresh()
    }

    var busy: Bool { state != .idle }

    func refresh() {
        downloaded = Set(ImageModels.catalog.filter(\.isDownloaded).map(\.id))
        let files = (try? FileManager.default.contentsOfDirectory(at: Self.outputs, includingPropertiesForKeys: nil)) ?? []
        gallery = files.filter { $0.pathExtension == "png" }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
            .map(GeneratedImage.init)
    }

    func download(_ m: ImageModel) {
        model = m
        state = .downloading(0)
        problem = nil
        Task {
            do {
                try await ImageModelDownloader().download(m) { p in
                    Task { @MainActor in self.state = .downloading(p) }
                }
                refresh()
            } catch {
                problem = "Download failed: \(error.localizedDescription)"
            }
            state = .idle
        }
    }

    func delete(_ m: ImageModel) {
        guard !busy else { return }
        if generator.loadedID == m.id { generator.unload() }
        try? FileManager.default.removeItem(at: m.directory)
        refresh()
    }

    func generate(_ prompt: String) {
        let prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !busy, model.isDownloaded else { return }
        problem = nil
        if model.cloud {
            generateCloud(prompt)
            return
        }
        if model.isXL { freeMemory?() }
        state = generator.loadedID == model.id ? .generating(0, steps) : .loading
        let (m, negative, steps, guidance) = (model, negative, steps, Float(guidance))
        let seed = UInt32.random(in: 0...UInt32.max)
        let generator = generator
        queue.async {
            let result = Result { () -> CGImage? in
                try generator.load(m)
                return try generator.generate(m, prompt: prompt, negative: negative, steps: steps,
                                              guidance: guidance, seed: seed) { step, total in
                    Task { @MainActor in self.state = .generating(step, total) }
                }
            }
            Task { @MainActor in
                switch result {
                case .success(let image?):
                    self.save(image)
                case .success(nil):
                    if !generator.cancelled { self.problem = "No image was produced" }
                case .failure(let e):
                    self.problem = e.localizedDescription
                }
                self.state = .idle
            }
        }
    }

    func cancel() {
        generator.cancelled = true
    }

    func remove(_ image: GeneratedImage) {
        try? FileManager.default.removeItem(at: image.url)
        if latest == image { latest = nil }
        refresh()
    }

    func save(_ image: CGImage) {
        try? FileManager.default.createDirectory(at: Self.outputs, withIntermediateDirectories: true)
        let name = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let url = Self.outputs.appendingPathComponent("\(name).png")
        if let data = UIImage(cgImage: image).pngData(), (try? data.write(to: url)) != nil {
            latest = GeneratedImage(url: url)
            UINotificationFeedbackGenerator().notificationOccurred(.success)
        }
        refresh()
    }
}
