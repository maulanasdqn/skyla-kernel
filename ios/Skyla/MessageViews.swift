import SwiftUI
import UIKit

struct MessagesView: View {
    @EnvironmentObject private var chat: ChatModel

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 20) {
                    if chat.messages.isEmpty {
                        WelcomeView()
                    }
                    ForEach(chat.messages) { message in
                        MessageRow(message: message).id(message.id)
                    }
                    Color.clear.frame(height: 4).id("bottom")
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
            }
            .defaultScrollAnchor(.bottom, for: .sizeChanges)
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: chat.messages.count) {
                withAnimation(.easeOut(duration: 0.25)) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
        }
    }
}

struct WelcomeView: View {
    @EnvironmentObject private var chat: ChatModel
    private let ideas = [
        "Jelaskan cara kerja LLM secara singkat",
        "Buatkan rencana belajar Rust 2 minggu",
        "Tulis fungsi Python untuk cek bilangan prima",
        "Ide nama untuk kedai kopi kecil",
    ]

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "sparkles")
                .font(.system(size: 40))
                .foregroundStyle(.tint)
                .padding(.top, 60)
            Text("Skyla Runtime")
                .font(.caption.weight(.semibold))
                .textCase(.uppercase)
                .kerning(1.5)
                .foregroundStyle(.secondary)
            Text(chat.current == nil ? "Pick a model to start" : "What's on your mind?")
                .font(.title2.weight(.semibold))
            Text("Runs fully on this iPhone. Nothing leaves the device.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            if let problem = chat.problem {
                Label(problem, systemImage: "exclamationmark.triangle.fill")
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .padding(.horizontal)
            }
            if chat.phase == .ready {
                VStack(spacing: 10) {
                    ForEach(ideas, id: \.self) { idea in
                        Button {
                            UIImpactFeedbackGenerator(style: .light).impactOccurred()
                            chat.send(idea)
                        } label: {
                            Text(idea)
                                .font(.subheadline)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 14)
                                .padding(.vertical, 12)
                                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14))
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.top, 8)
            }
        }
        .frame(maxWidth: .infinity)
        .multilineTextAlignment(.center)
    }
}

struct MessageRow: View {
    @ObservedObject var message: MessageModel

    var body: some View {
        if message.user {
            HStack {
                Spacer(minLength: 48)
                Text(message.text)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                    .contextMenu { CopyButton(text: message.text) }
            }
            .transition(.move(edge: .bottom).combined(with: .opacity))
        } else {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "sparkles")
                    .font(.footnote)
                    .foregroundStyle(.tint)
                    .frame(width: 26, height: 26)
                    .background(Color.accentColor.opacity(0.12), in: Circle())
                VStack(alignment: .leading, spacing: 10) {
                    if !message.thinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        ThinkingView(message: message)
                    }
                    if message.text.isEmpty && message.streaming && message.error == nil {
                        if message.thinking.isEmpty { TypingDots() }
                    } else {
                        MarkdownView(text: message.text)
                            .textSelection(.enabled)
                            .contextMenu { CopyButton(text: message.text) }
                    }
                    if let error = message.error {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                    if let stats = message.stats, !message.streaming {
                        Text(stats)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                            .transition(.opacity)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .animation(.easeOut(duration: 0.2), value: message.streaming)
        }
    }
}

struct ThinkingView: View {
    @ObservedObject var message: MessageModel
    @State private var expanded: Bool?

    var body: some View {
        let open = expanded ?? !message.answerStarted
        VStack(alignment: .leading, spacing: 6) {
            Button {
                withAnimation(.easeInOut(duration: 0.2)) { expanded = !open }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "brain")
                    Text(message.answerStarted || !message.streaming ? "Thought process" : "Thinking…")
                    Image(systemName: "chevron.right")
                        .rotationEffect(.degrees(open ? 90 : 0))
                }
                .font(.footnote.weight(.medium))
                .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            if open {
                Text(message.thinking.trimmingCharacters(in: .whitespacesAndNewlines))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .padding(.leading, 10)
                    .overlay(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 1).fill(.quaternary).frame(width: 2)
                    }
            }
        }
    }
}

struct TypingDots: View {
    var body: some View {
        TimelineView(.animation) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            HStack(spacing: 5) {
                ForEach(0..<3) { i in
                    Circle()
                        .frame(width: 7, height: 7)
                        .opacity(0.3 + 0.7 * max(0, sin(t * 5 - Double(i) * 0.8)))
                }
            }
            .foregroundStyle(.secondary)
            .padding(.vertical, 8)
        }
    }
}

struct CopyButton: View {
    let text: String

    var body: some View {
        Button("Copy", systemImage: "doc.on.doc") { UIPasteboard.general.string = text }
    }
}
