import CoreML
import Foundation
import ImageIO
import StableDiffusion
import UniformTypeIdentifiers

struct Job: Decodable {
    let resources: String
    let xl: Bool
    let prompt: String
    let negative: String
    let steps: Int
    let guidance: Float
    let seed: UInt32
    let output: String
}

struct Command: Decodable {
    let cancel: Bool?
    let job: Job?
}

func emit(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: object),
          let line = String(data: data, encoding: .utf8) else { return }
    print(line)
    fflush(stdout)
}

final class Generator: @unchecked Sendable {
    private var pipeline: (any StableDiffusionPipelineProtocol)?
    private var loaded: String?
    var cancelled = false

    func run(_ job: Job) {
        let started = Date()
        do {
            if loaded != job.resources {
                emit(["type": "loading"])
                pipeline = nil
                let config = MLModelConfiguration()
                config.computeUnits = .cpuAndGPU
                let url = URL(fileURLWithPath: job.resources)
                let p: any StableDiffusionPipelineProtocol = job.xl
                    ? try StableDiffusionXLPipeline(resourcesAt: url, configuration: config, reduceMemory: false)
                    : try StableDiffusionPipeline(resourcesAt: url, controlNet: [], configuration: config,
                                                  disableSafety: true, reduceMemory: false)
                try p.loadResources()
                pipeline = p
                loaded = job.resources
            }
            guard let pipeline else { return }
            cancelled = false
            var config = PipelineConfiguration(prompt: job.prompt)
            config.negativePrompt = job.negative
            config.stepCount = job.steps
            config.guidanceScale = job.guidance
            config.seed = job.seed
            config.disableSafety = true
            config.schedulerType = .dpmSolverMultistepScheduler
            config.schedulerTimestepSpacing = .karras
            let scale: Float32 = job.xl ? 0.13025 : 0.18215
            config.encoderScaleFactor = scale
            config.decoderScaleFactor = scale
            let sampling = Date()
            let images = try pipeline.generateImages(configuration: config) { p in
                emit(["type": "step", "step": p.step + 1, "total": p.stepCount])
                return !self.cancelled
            }
            guard let image = images.first ?? nil else {
                emit(["type": cancelled ? "cancelled" : "error", "message": "no image was produced"])
                return
            }
            let url = URL(fileURLWithPath: job.output)
            guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
                throw CocoaError(.fileWriteUnknown)
            }
            CGImageDestinationAddImage(dest, image, nil)
            guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
            emit([
                "type": "done", "output": job.output,
                "seconds": Date().timeIntervalSince(started),
                "sampling_seconds": Date().timeIntervalSince(sampling),
            ])
        } catch {
            emit(["type": "error", "message": "\(error)"])
        }
    }
}

let generator = Generator()
let queue = DispatchQueue(label: "skyla.image")
emit(["type": "ready"])
while let line = readLine() {
    guard let data = line.data(using: .utf8),
          let command = try? JSONDecoder().decode(Command.self, from: data) else { continue }
    if command.cancel == true {
        generator.cancelled = true
    } else if let job = command.job {
        queue.async { generator.run(job) }
    }
}
queue.sync {}
