import Foundation
import Security

enum Keychain {
    private static let service = "com.maulanasdqn.skyla.fal"

    static func get() -> String {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                    kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return "" }
        return String(data: data, encoding: .utf8) ?? ""
    }

    static func set(_ value: String) {
        let base: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service]
        SecItemDelete(base as CFDictionary)
        guard !value.isEmpty else { return }
        var add = base
        add[kSecValueData as String] = Data(value.utf8)
        SecItemAdd(add as CFDictionary, nil)
    }
}

struct FalError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

enum FalClient {
    static func body(model: String, prompt: String, steps: Int, guidance: Double) -> [String: Any] {
        let seed = Int.random(in: 0..<Int(Int32.max))
        if model.contains("flux-pro") {
            return ["prompt": prompt, "aspect_ratio": "1:1", "safety_tolerance": "5",
                    "enable_safety_checker": false, "output_format": "png", "seed": seed]
        }
        let steps = model.hasSuffix("schnell") ? min(max(steps, 1), 12) : min(max(steps, 4), 50)
        return ["prompt": prompt, "image_size": "square_hd", "num_inference_steps": steps,
                "guidance_scale": min(max(guidance, 1), 10), "enable_safety_checker": false,
                "output_format": "png", "seed": seed]
    }

    static func generate(model: String, prompt: String, steps: Int, guidance: Double, key: String)
        async throws -> (Data, flagged: Bool) {
        var request = URLRequest(url: URL(string: "https://fal.run/\(model)")!)
        request.httpMethod = "POST"
        request.setValue("Key \(key)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 300
        request.httpBody = try JSONSerialization.data(withJSONObject: body(model: model, prompt: prompt,
                                                                           steps: steps, guidance: guidance))
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode >= 400 {
            let text = String(data: data, encoding: .utf8) ?? ""
            throw FalError(message: "fal.ai returned \(http.statusCode): \(text.prefix(200))")
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let images = json["images"] as? [[String: Any]],
              let link = images.first?["url"] as? String, let url = URL(string: link) else {
            throw FalError(message: "fal.ai returned no image")
        }
        let (image, _) = try await URLSession.shared.data(from: url)
        let flagged = (json["has_nsfw_concepts"] as? [Bool])?.contains(true) ?? false
        return (image, flagged)
    }
}
