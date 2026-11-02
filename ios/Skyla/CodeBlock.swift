import SwiftUI
import UIKit

struct CodeBlock: View {
    let language: String
    let code: String
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language.isEmpty ? "code" : language.lowercased())
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    UIPasteboard.general.string = code
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    withAnimation { copied = true }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { withAnimation { copied = false } }
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(Color(.tertiarySystemFill))
            ScrollView(.horizontal, showsIndicators: false) {
                Text(Highlighter.highlight(code, language: language))
                    .font(.system(.footnote, design: .monospaced))
                    .lineSpacing(2)
                    .textSelection(.enabled)
                    .padding(12)
            }
        }
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(Color(.separator), lineWidth: 0.5))
    }
}

enum Highlighter {
    private static let keywords: Set<String> = [
        "as", "async", "await", "break", "case", "catch", "class", "const", "continue", "def", "default",
        "do", "elif", "else", "enum", "export", "extends", "false", "fn", "for", "from", "func", "function",
        "guard", "if", "impl", "import", "in", "interface", "let", "loop", "match", "mut", "new", "nil",
        "None", "null", "pass", "private", "pub", "public", "return", "self", "Self", "static", "struct",
        "super", "switch", "throw", "trait", "true", "True", "False", "try", "type", "use", "var", "where",
        "while", "with", "yield", "lambda", "package", "go", "defer", "int", "string", "bool", "void",
    ]

    static func highlight(_ code: String, language: String) -> AttributedString {
        let hashComments = ["python", "py", "sh", "bash", "zsh", "ruby", "rb", "yaml", "yml", "toml"]
            .contains(language.lowercased())
        var out = AttributedString()
        let chars = Array(code)
        var i = 0
        func emit(_ s: String, _ color: Color?) {
            var part = AttributedString(s)
            if let color { part.foregroundColor = color }
            out += part
        }
        while i < chars.count {
            let c = chars[i]
            let next = i + 1 < chars.count ? chars[i + 1] : " "
            if (c == "/" && next == "/") || (hashComments && c == "#") {
                var j = i
                while j < chars.count && chars[j] != "\n" { j += 1 }
                emit(String(chars[i..<j]), .green)
                i = j
            } else if c == "\"" || c == "'" || c == "`" {
                var j = i + 1
                while j < chars.count && chars[j] != c && chars[j] != "\n" {
                    j += chars[j] == "\\" ? 2 : 1
                }
                j = min(j + 1, chars.count)
                emit(String(chars[i..<j]), .orange)
                i = j
            } else if c.isNumber {
                var j = i
                while j < chars.count && (chars[j].isNumber || chars[j] == "." || chars[j] == "_") { j += 1 }
                emit(String(chars[i..<j]), .blue)
                i = j
            } else if c.isLetter || c == "_" {
                var j = i
                while j < chars.count && (chars[j].isLetter || chars[j].isNumber || chars[j] == "_") { j += 1 }
                let word = String(chars[i..<j])
                emit(word, keywords.contains(word) ? .purple : nil)
                i = j
            } else {
                emit(String(c), nil)
                i += 1
            }
        }
        return out
    }
}
