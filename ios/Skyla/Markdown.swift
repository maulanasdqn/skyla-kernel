import SwiftUI
import UIKit

enum ListMarker: Hashable {
    case bullet
    case number(String)
    case task(Bool)
}

enum Block: Hashable {
    case heading(Int, String)
    case paragraph(String)
    case item(Int, ListMarker, String)
    case quote(String)
    case code(String, String)
    case table([String], [[String]])
    case rule
}

enum Markdown {
    static func blocks(_ text: String) -> [Block] {
        var out: [Block] = []
        var paragraph: [String] = []
        var quote: [String] = []
        var table: [String] = []
        var code: [String]?
        var language = ""

        func flush() {
            if !paragraph.isEmpty { out.append(.paragraph(paragraph.joined(separator: "\n"))) }
            if !quote.isEmpty { out.append(.quote(quote.joined(separator: "\n"))) }
            if !table.isEmpty { out.append(contentsOf: Self.table(table)) }
            paragraph = []
            quote = []
            table = []
        }

        for raw in text.components(separatedBy: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("```") {
                if let lines = code {
                    out.append(.code(language, lines.joined(separator: "\n")))
                    code = nil
                } else {
                    flush()
                    language = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                    code = []
                }
                continue
            }
            if code != nil {
                code?.append(raw)
                continue
            }
            let indent = raw.prefix(while: { $0 == " " || $0 == "\t" }).count / 2
            if line.hasPrefix("|") {
                if table.isEmpty { flush() }
                table.append(line)
                continue
            } else if !table.isEmpty {
                flush()
            }
            if line.isEmpty {
                flush()
            } else if let level = heading(line) {
                flush()
                out.append(.heading(level, String(line.drop(while: { $0 == "#" })).trimmingCharacters(in: .whitespaces)))
            } else if line == "---" || line == "***" || line == "___" {
                flush()
                out.append(.rule)
            } else if let (marker, rest) = item(line) {
                flush()
                out.append(.item(indent, marker, rest))
            } else if line.hasPrefix(">") {
                if !paragraph.isEmpty { flush() }
                quote.append(String(line.dropFirst()).trimmingCharacters(in: .whitespaces))
            } else {
                if !quote.isEmpty { flush() }
                paragraph.append(line)
            }
        }
        if let lines = code { out.append(.code(language, lines.joined(separator: "\n"))) }
        flush()
        return out
    }

    private static func heading(_ line: String) -> Int? {
        let hashes = line.prefix(while: { $0 == "#" }).count
        guard (1...6).contains(hashes), line.dropFirst(hashes).first == " " else { return nil }
        return hashes
    }

    private static func item(_ line: String) -> (ListMarker, String)? {
        for bullet in ["- ", "* ", "+ ", "• "] where line.hasPrefix(bullet) {
            let rest = String(line.dropFirst(2))
            if rest.hasPrefix("[ ] ") { return (.task(false), String(rest.dropFirst(4))) }
            if rest.lowercased().hasPrefix("[x] ") { return (.task(true), String(rest.dropFirst(4))) }
            return (.bullet, rest)
        }
        let digits = line.prefix(while: \.isNumber)
        guard !digits.isEmpty, digits.count <= 3 else { return nil }
        let rest = line.dropFirst(digits.count)
        guard rest.hasPrefix(". ") || rest.hasPrefix(") ") else { return nil }
        return (.number(String(digits)), String(rest.dropFirst(2)))
    }

    private static func cells(_ line: String) -> [String] {
        var parts = line.split(separator: "|", omittingEmptySubsequences: false).map {
            $0.trimmingCharacters(in: .whitespaces)
        }
        if parts.first == "" { parts.removeFirst() }
        if parts.last == "" { parts.removeLast() }
        return parts
    }

    private static func table(_ lines: [String]) -> [Block] {
        let rows = lines.map(cells)
        let isRule = { (row: [String]) in
            !row.isEmpty && row.allSatisfy { cell in !cell.isEmpty && cell.allSatisfy { "-: ".contains($0) } }
        }
        guard rows.count >= 2, isRule(rows[1]) else {
            return [.paragraph(lines.joined(separator: "\n"))]
        }
        return [.table(rows[0], rows.dropFirst(2).filter { !isRule($0) })]
    }

    static func inline(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        var result = (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
        for run in result.runs {
            if let intent = run.inlinePresentationIntent, intent.contains(.code) {
                result[run.range].font = .system(.callout, design: .monospaced)
                result[run.range].backgroundColor = Color(.tertiarySystemFill)
                result[run.range].foregroundColor = .pink
            }
            if run.link != nil {
                result[run.range].foregroundColor = .accentColor
                result[run.range].underlineStyle = .single
            }
        }
        return result
    }
}
