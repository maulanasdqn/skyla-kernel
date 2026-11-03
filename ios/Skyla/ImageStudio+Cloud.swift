import Foundation
import UIKit

extension ImageStudio {
    func generateCloud(_ prompt: String) {
        let key = falKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else {
            problem = "Add your fal.ai API key under Advanced to use cloud models."
            return
        }
        state = .remote
        let (model, steps, guidance) = (model.id, steps, guidance)
        Task {
            do {
                let (data, flagged) = try await FalClient.generate(model: model, prompt: prompt, steps: steps,
                                                                   guidance: guidance, key: key)
                if let image = UIImage(data: data)?.cgImage {
                    save(image)
                }
                if flagged {
                    problem = "fal.ai flagged this image as unsafe and blacked it out. Turning its safety checker "
                        + "off requires fal.ai to authorize your account."
                }
            } catch {
                problem = error.localizedDescription
            }
            state = .idle
        }
    }
}
