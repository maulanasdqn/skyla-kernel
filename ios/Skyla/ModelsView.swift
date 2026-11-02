import SwiftUI
import UIKit

struct ModelsView: View {
    @EnvironmentObject private var chat: ChatModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section { ChipCard() }
                    .listRowInsets(EdgeInsets())
                    .listRowBackground(Color.clear)
                Section {
                    ForEach(chat.models) { option in
                        Button {
                            chat.select(option)
                            dismiss()
                        } label: { ModelRow(option: option, active: chat.current == option) }
                            .buttonStyle(.plain)
                            .disabled(chat.busy)
                            .swipeActions {
                                if option.isDownloaded && chat.current != option {
                                    Button("Delete", systemImage: "trash", role: .destructive) { chat.delete(option) }
                                }
                            }
                    }
                } header: {
                    Text("Models")
                } footer: {
                    Text("Add your own .gguf files to the Skyla Runtime folder in the Files app. Swipe left to delete a downloaded model.")
                }
            }
            .navigationTitle("Choose a model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { Button("Done") { dismiss() } }
            .onAppear { chat.refresh() }
        }
    }
}

struct ChipCard: View {
    var body: some View {
        HStack(spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(LinearGradient(colors: [.indigo, .purple], startPoint: .topLeading, endPoint: .bottomTrailing))
                Image(systemName: "cpu")
                    .font(.title2.weight(.semibold))
                    .foregroundStyle(.white)
            }
            .frame(width: 52, height: 52)
            VStack(alignment: .leading, spacing: 4) {
                Text(ChatModel.chip).font(.title3.weight(.bold))
                Text("Metal 4 · Neural Accelerators")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Label("Runs fully on this iPhone", systemImage: "lock.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
            }
            Spacer(minLength: 0)
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }
}

struct ModelRow: View {
    let option: ModelOption
    let active: Bool

    private var size: String {
        let bytes = option.isDownloaded ? Double(option.bytesOnDisk) : option.sizeGB * 1e9
        return String(format: "%.1f GB", bytes / 1e9)
    }

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: active ? "checkmark" : "brain")
                .font(.body.weight(.semibold))
                .foregroundStyle(active ? .white : Color.accentColor)
                .frame(width: 36, height: 36)
                .background(active ? Color.green : Color.accentColor.opacity(0.12), in: Circle())
            VStack(alignment: .leading, spacing: 3) {
                Text(option.name).font(.subheadline.weight(.semibold)).lineLimit(2)
                Text(option.detail).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            Spacer(minLength: 8)
            Badge(text: active ? "Active" : option.isDownloaded ? size : "↓ \(size)",
                  color: active ? .green : option.isDownloaded ? .secondary : .accentColor)
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }
}

struct Badge: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .monospacedDigit()
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(color.opacity(0.12), in: Capsule())
    }
}
