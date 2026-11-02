import SwiftUI
import UIKit

struct SettingsView: View {
    @EnvironmentObject private var chat: ChatModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Sampling") {
                    VStack(alignment: .leading) {
                        Text("Temperature \(chat.temperature, specifier: "%.2f")")
                        Slider(value: $chat.temperature, in: 0...1.5, step: 0.05)
                    }
                    Stepper("Max tokens \(chat.maxTokens)", value: $chat.maxTokens, in: 128...4096, step: 128)
                }
                Section("System prompt") {
                    TextField("Optional persona or instructions", text: $chat.system, axis: .vertical)
                        .lineLimit(3...8)
                }
                Section {
                    Text("Context: \(ChatModel.context) tokens. Put your own .gguf files in the Skyla folder of the Files app to load them.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { Button("Done") { dismiss() } }
        }
    }
}
