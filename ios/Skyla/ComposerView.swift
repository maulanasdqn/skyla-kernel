import SwiftUI
import UIKit

struct ComposerView: View {
    @EnvironmentObject private var chat: ChatModel
    @State private var draft = ""
    @FocusState private var focused: Bool

    private var canSend: Bool {
        chat.phase == .ready && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField(chat.current == nil ? "Load a model first" : "Message", text: $draft, axis: .vertical)
                .focused($focused)
                .lineLimit(1...6)
                .padding(.horizontal, 16)
                .padding(.vertical, 11)
                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            Group {
                if chat.phase == .generating {
                    Button {
                        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                        chat.stop()
                    } label: {
                        Image(systemName: "stop.fill")
                            .font(.system(size: 14, weight: .bold))
                            .frame(width: 40, height: 40)
                            .background(Color.primary, in: Circle())
                            .foregroundStyle(Color(.systemBackground))
                    }
                } else {
                    Button {
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()
                        chat.send(draft)
                        draft = ""
                    } label: {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 17, weight: .bold))
                            .frame(width: 40, height: 40)
                            .background(canSend ? Color.accentColor : Color(.tertiarySystemFill), in: Circle())
                            .foregroundStyle(canSend ? .white : .secondary)
                    }
                    .disabled(!canSend)
                }
            }
            .animation(.easeInOut(duration: 0.15), value: chat.phase)
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .padding(.bottom, 10)
        .background(.bar)
    }
}
