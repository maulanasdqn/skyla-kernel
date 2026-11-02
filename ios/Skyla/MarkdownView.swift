import SwiftUI
import UIKit

struct MarkdownView: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(Markdown.blocks(text).enumerated()), id: \.offset) { _, block in
                BlockView(block: block)
            }
        }
        .tint(.accentColor)
    }
}

struct BlockView: View {
    let block: Block

    var body: some View {
        switch block {
        case .heading(let level, let text):
            Text(Markdown.inline(text))
                .font(font(level))
                .padding(.top, level <= 2 ? 6 : 2)
        case .paragraph(let text):
            Text(Markdown.inline(text))
                .lineSpacing(3)
                .fixedSize(horizontal: false, vertical: true)
        case .item(let indent, let marker, let text):
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                markerView(marker)
                Text(Markdown.inline(text))
                    .lineSpacing(3)
                    .strikethrough(marker == .task(true), color: .secondary)
                    .foregroundStyle(marker == .task(true) ? .secondary : .primary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.leading, CGFloat(indent) * 18)
        case .quote(let text):
            HStack(spacing: 12) {
                RoundedRectangle(cornerRadius: 2).fill(Color.accentColor.opacity(0.5)).frame(width: 3)
                Text(Markdown.inline(text))
                    .italic()
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.vertical, 2)
        case .code(let language, let code):
            CodeBlock(language: language, code: code)
        case .table(let header, let rows):
            TableBlock(header: header, rows: rows)
        case .rule:
            Divider().padding(.vertical, 4)
        }
    }

    private func font(_ level: Int) -> Font {
        switch level {
        case 1: .title2.bold()
        case 2: .title3.bold()
        case 3: .headline
        default: .subheadline.bold()
        }
    }

    @ViewBuilder
    private func markerView(_ marker: ListMarker) -> some View {
        switch marker {
        case .bullet:
            Circle().fill(.secondary).frame(width: 5, height: 5).offset(y: -3)
        case .number(let n):
            Text("\(n).").foregroundStyle(.secondary).monospacedDigit()
        case .task(let done):
            Image(systemName: done ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(done ? Color.green : Color.secondary)
                .font(.callout)
        }
    }
}

struct TableBlock: View {
    let header: [String]
    let rows: [[String]]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
                GridRow {
                    ForEach(Array(header.enumerated()), id: \.offset) { _, cell in
                        Text(Markdown.inline(cell))
                            .font(.subheadline.bold())
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                    }
                }
                .background(Color(.tertiarySystemFill))
                ForEach(Array(rows.enumerated()), id: \.offset) { index, row in
                    GridRow {
                        ForEach(0..<header.count, id: \.self) { col in
                            Text(Markdown.inline(col < row.count ? row[col] : ""))
                                .font(.subheadline)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                        }
                    }
                    .background(index % 2 == 1 ? Color(.secondarySystemBackground) : Color.clear)
                }
            }
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color(.separator), lineWidth: 0.5))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }
}
