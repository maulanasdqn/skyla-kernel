import Foundation
import SwiftUI

final class PendingText: @unchecked Sendable {
    private let lock = NSLock()
    private var answer = ""
    private var thinking = ""
    private var finished = false

    func append(_ piece: String, thinking isThinking: Bool) {
        lock.lock()
        if isThinking { thinking += piece } else { answer += piece }
        lock.unlock()
    }

    func finish() {
        lock.lock()
        finished = true
        lock.unlock()
    }

    func take() -> (answer: String, thinking: String, finished: Bool) {
        lock.lock()
        defer {
            answer = ""
            thinking = ""
            lock.unlock()
        }
        return (answer, thinking, finished)
    }
}

@MainActor
final class MessageModel: ObservableObject, Identifiable {
    let id = UUID()
    let user: Bool
    @Published private(set) var text: String
    @Published private(set) var thinking = ""
    @Published var stats: String?
    @Published var error: String?
    @Published private(set) var streaming: Bool

    private var target = ""
    private var targetThinking = ""

    init(user: Bool, text: String = "", streaming: Bool = false) {
        self.user = user
        self.text = text
        self.streaming = streaming
    }

    var answerStarted: Bool { !text.isEmpty }

    func receive(_ pending: (answer: String, thinking: String, finished: Bool)) {
        target += pending.answer
        targetThinking += pending.thinking
    }

    func reveal(finished: Bool) -> Bool {
        thinking = advance(thinking, toward: targetThinking)
        text = advance(text, toward: target)
        let caughtUp = text.count == target.count && thinking.count == targetThinking.count
        if finished && caughtUp {
            streaming = false
        }
        return streaming
    }

    private func advance(_ shown: String, toward target: String) -> String {
        let remaining = target.count - shown.count
        guard remaining > 0 else { return shown }
        let step = max(1, remaining / 6)
        return String(target.prefix(shown.count + step))
    }
}
