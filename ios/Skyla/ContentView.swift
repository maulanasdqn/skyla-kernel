import SwiftUI
import UIKit

struct ContentView: View {
    @EnvironmentObject private var chat: ChatModel
    @State private var showModels = false
    @State private var showSettings = false

    var body: some View {
        NavigationStack {
            MessagesView()
                .safeAreaInset(edge: .bottom, spacing: 0) { ComposerView() }
                .safeAreaInset(edge: .top, spacing: 0) { DownloadLine() }
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .principal) {
                        ModelHeader { showModels = true }
                    }
                    .sharedBackgroundVisibility(.hidden)
                    ToolbarItem(placement: .topBarLeading) {
                        Button {
                            UIImpactFeedbackGenerator(style: .light).impactOccurred()
                            chat.newChat()
                        } label: { Image(systemName: "square.and.pencil") }
                            .disabled(chat.phase != .ready || chat.messages.isEmpty)
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        Menu {
                            Button("Models", systemImage: "square.stack.3d.up") { showModels = true }
                            Button("Settings", systemImage: "slider.horizontal.3") { showSettings = true }
                            Divider()
                            Button("Flush memory", systemImage: "memorychip", role: .destructive) { chat.flush() }
                                .disabled(chat.phase != .ready)
                        } label: { Image(systemName: "ellipsis") }
                    }
                }
                .sheet(isPresented: $showModels) { ModelsView() }
                .sheet(isPresented: $showSettings) { SettingsView() }
                .onAppear { if chat.current == nil && !chat.busy { showModels = true } }
        }
    }
}

struct ModelHeader: View {
    @EnvironmentObject private var chat: ChatModel
    let action: () -> Void

    private var dot: Color {
        switch chat.phase {
        case .ready: .green
        case .generating: .blue
        case .loading, .downloading: .orange
        case .idle: chat.problem == nil ? Color(.systemGray3) : .red
        }
    }

    private var subtitle: String {
        let context = "\(ChatModel.context / 1024)K context"
        switch chat.phase {
        case .idle: return chat.problem == nil ? "\(ChatModel.chip) · no model loaded" : "Something went wrong"
        case .downloading(let p): return "Downloading · \(Int(p * 100))%"
        case .loading: return "Loading onto \(ChatModel.chip)…"
        case .ready: return "\(ChatModel.chip) · \(context)"
        case .generating: return "\(ChatModel.chip) · writing…"
        }
    }

    var body: some View {
        Button(action: action) {
            VStack(spacing: 2) {
                HStack(spacing: 4) {
                    Text(chat.current?.short ?? "Skyla")
                        .font(.headline)
                        .lineLimit(1)
                    Image(systemName: "chevron.down")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.secondary)
                }
                HStack(spacing: 5) {
                    Circle()
                        .fill(dot)
                        .frame(width: 6, height: 6)
                        .shadow(color: dot.opacity(0.6), radius: 3)
                    Text(subtitle)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                        .lineLimit(1)
                        .contentTransition(.numericText())
                }
                .animation(.easeInOut(duration: 0.2), value: subtitle)
            }
            .padding(.horizontal, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(chat.busy && chat.phase != .ready)
    }
}

struct DownloadLine: View {
    @EnvironmentObject private var chat: ChatModel

    var body: some View {
        if case .downloading(let p) = chat.phase {
            GeometryReader { geo in
                Capsule()
                    .fill(Color.accentColor)
                    .frame(width: max(4, geo.size.width * p))
                    .animation(.linear(duration: 0.2), value: p)
            }
            .frame(height: 3)
            .padding(.horizontal, 16)
            .padding(.top, 2)
        }
    }
}
