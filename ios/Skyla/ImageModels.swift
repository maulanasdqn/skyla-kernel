import Foundation

struct ImageModel: Identifiable, Hashable {
    let id: String
    let name: String
    let detail: String
    let repo: String
    let folder: String
    let sizeGB: Double
    let isXL: Bool
    let defaultSteps: Int
    var cloud = false

    var directory: URL { ImageModels.directory.appendingPathComponent(id, isDirectory: true) }
    var isDownloaded: Bool {
        cloud || FileManager.default.fileExists(atPath: directory.appendingPathComponent(".complete").path)
    }
}

enum ImageModels {
    static let directory = Models.directory.appendingPathComponent("ImageModels", isDirectory: true)
    static let skipped = ["SafetyChecker.mlmodelc", "VAEEncoder.mlmodelc"]

    static let catalog: [ImageModel] = [
        ImageModel(
            id: "sd15", name: "Stable Diffusion 1.5", detail: "Fast · 512 × 512 · huge community style range",
            repo: "apple/coreml-stable-diffusion-v1-5-palettized", folder: "split_einsum_v2/compiled",
            sizeGB: 0.95, isXL: false, defaultSteps: 20),
        ImageModel(
            id: "sdxl", name: "Stable Diffusion XL", detail: "Best quality · 768 × 768 · slower",
            repo: "apple/coreml-stable-diffusion-xl-base-ios",
            folder: "coreml-stable-diffusion-xl-base-ios_split_einsum_compiled",
            sizeGB: 3.05, isXL: true, defaultSteps: 20),
        cloudModel("fal-ai/flux/schnell", "FLUX schnell", "Cloud · fal.ai · fastest", steps: 4),
        cloudModel("fal-ai/flux/dev", "FLUX dev", "Cloud · fal.ai · detailed", steps: 28),
        cloudModel("fal-ai/flux-2-dev", "FLUX.2 dev", "Cloud · fal.ai · newest", steps: 28),
        cloudModel("fal-ai/flux-pro/v1.1-ultra", "FLUX pro ultra", "Cloud · fal.ai · top quality", steps: 28),
    ]

    private static func cloudModel(_ id: String, _ name: String, _ detail: String, steps: Int) -> ImageModel {
        ImageModel(id: id, name: name, detail: detail, repo: "", folder: "", sizeGB: 0, isXL: true,
                   defaultSteps: steps, cloud: true)
    }
}

private struct TreeEntry: Decodable {
    let type: String
    let path: String
    let size: Int64?
}

final class ImageModelDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private var done: Int64 = 0
    private var total: Int64 = 1
    private var report: ((Double) -> Void)?
    private var target: URL?
    private var moveError: Error?
    private var finished: CheckedContinuation<Void, Error>?

    func download(_ model: ImageModel, progress: @escaping (Double) -> Void) async throws {
        report = progress
        let api = URL(string: "https://huggingface.co/api/models/\(model.repo)/tree/main/\(model.folder)?recursive=1")!
        let (data, _) = try await URLSession.shared.data(from: api)
        let files = try JSONDecoder().decode([TreeEntry].self, from: data).filter { entry in
            entry.type == "file" && !ImageModels.skipped.contains { entry.path.contains($0) }
        }
        total = max(1, files.reduce(0) { $0 + ($1.size ?? 0) })
        done = 0
        let fm = FileManager.default
        try? fm.removeItem(at: model.directory)
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        for file in files {
            let relative = String(file.path.dropFirst(model.folder.count + 1))
            let destination = model.directory.appendingPathComponent(relative)
            try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            target = destination
            moveError = nil
            let source = URL(string: "https://huggingface.co/\(model.repo)/resolve/main/\(file.path)")!
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                finished = c
                session.downloadTask(with: source).resume()
            }
            done += file.size ?? 0
            progress(Double(done) / Double(total))
        }
        fm.createFile(atPath: model.directory.appendingPathComponent(".complete").path, contents: Data())
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData _: Int64,
                    totalBytesWritten written: Int64, totalBytesExpectedToWrite _: Int64) {
        report?(Double(done + written) / Double(total))
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard let target else { return }
        do {
            try? FileManager.default.removeItem(at: target)
            try FileManager.default.moveItem(at: location, to: target)
        } catch {
            moveError = error
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let c = finished
        finished = nil
        if let error = error ?? moveError {
            c?.resume(throwing: error)
        } else if let http = task.response as? HTTPURLResponse, http.statusCode >= 400 {
            c?.resume(throwing: URLError(.badServerResponse))
        } else {
            c?.resume()
        }
    }
}
