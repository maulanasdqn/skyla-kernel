import Foundation

struct ModelOption: Identifiable, Hashable {
    let id: String
    let name: String
    let short: String
    let detail: String
    let url: URL?
    let sizeGB: Double

    var file: URL { Models.directory.appendingPathComponent(id) }
    var isDownloaded: Bool { FileManager.default.fileExists(atPath: file.path) }
    var bytesOnDisk: Int64 {
        (try? FileManager.default.attributesOfItem(atPath: file.path)[.size] as? Int64) ?? 0
    }
}

enum Models {
    static let directory: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

    static let catalog: [ModelOption] = [
        ModelOption(
            id: "qwen3-4b-2507-abliterated-q4_k_m.gguf",
            name: "Qwen3 4B Instruct 2507 Abliterated",
            short: "Qwen3 4B",
            detail: "Uncensored, no thinking, fast replies",
            url: URL(string: "https://huggingface.co/mradermacher/Huihui-Qwen3-4B-Instruct-2507-abliterated-GGUF/resolve/main/Huihui-Qwen3-4B-Instruct-2507-abliterated.Q4_K_M.gguf"),
            sizeGB: 2.5),
        ModelOption(
            id: "qwen3-0.6b-q8_0.gguf",
            name: "Qwen3 0.6B",
            short: "Qwen3 0.6B",
            detail: "Tiny and quick, thinks before answering",
            url: URL(string: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"),
            sizeGB: 0.64),
    ]

    static func all() -> [ModelOption] {
        let known = Set(catalog.map(\.id))
        let files = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        let imported = files.filter { $0.hasSuffix(".gguf") && !known.contains($0) }.sorted().map { name in
            ModelOption(id: name, name: name, short: String(name.dropLast(5)), detail: "Imported through Files",
                        url: nil, sizeGB: 0)
        }
        return catalog + imported
    }
}

final class Downloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private var progress: ((Double) -> Void)?
    private var finish: ((Result<URL, Error>) -> Void)?
    private var destination: URL?
    private var task: URLSessionDownloadTask?

    func start(_ option: ModelOption, progress: @escaping (Double) -> Void,
               finish: @escaping (Result<URL, Error>) -> Void) {
        guard let url = option.url else { return }
        self.progress = progress
        self.finish = finish
        destination = option.file
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        task = session.downloadTask(with: url)
        task?.resume()
    }

    func cancel() { task?.cancel() }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData _: Int64,
                    totalBytesWritten written: Int64, totalBytesExpectedToWrite expected: Int64) {
        if expected > 0 { progress?(Double(written) / Double(expected)) }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard let destination else { return }
        do {
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: location, to: destination)
            finish?(.success(destination))
        } catch {
            finish?(.failure(error))
        }
        session.finishTasksAndInvalidate()
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error { finish?(.failure(error)) }
        session.finishTasksAndInvalidate()
    }
}
