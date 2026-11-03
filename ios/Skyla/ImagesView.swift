import SwiftUI
import UIKit

struct ImagesView: View {
    @EnvironmentObject private var studio: ImageStudio
    @State private var prompt = ""
    @State private var viewing: GeneratedImage?
    @FocusState private var focused: Bool

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    ImageModelPicker()
                    PromptCard(prompt: $prompt, focused: $focused)
                    if let problem = studio.problem {
                        Label(problem, systemImage: "exclamationmark.triangle.fill")
                            .font(.footnote)
                            .foregroundStyle(.red)
                    }
                    if let latest = studio.latest {
                        ResultCard(image: latest) { viewing = latest }
                    }
                    GalleryGrid { viewing = $0 }
                }
                .padding(16)
            }
            .scrollDismissesKeyboard(.interactively)
            .navigationTitle("Images")
            .sheet(item: $viewing) { ImageViewer(image: $0) }
            .onAppear { studio.refresh() }
        }
    }
}

struct ImageModelPicker: View {
    @EnvironmentObject private var studio: ImageStudio

    var body: some View {
        LazyVGrid(columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)], spacing: 10) {
            ForEach(ImageModels.catalog) { m in
                let selected = studio.model == m
                let ready = studio.downloaded.contains(m.id)
                Button {
                    if ready { studio.model = m } else if !studio.busy { studio.download(m) }
                } label: {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(m.cloud ? m.name : m.isXL ? "SDXL" : "SD 1.5").font(.headline).lineLimit(1)
                            if m.cloud { Image(systemName: "cloud.fill").font(.caption).foregroundStyle(.secondary) }
                            Spacer()
                            if ready {
                                Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                                    .foregroundStyle(selected ? Color.accentColor : .secondary)
                            }
                        }
                        Text(m.detail).font(.caption2).foregroundStyle(.secondary).lineLimit(2)
                        if case .downloading(let p) = studio.state, studio.model == m, !ready {
                            ProgressView(value: p).tint(.accentColor)
                            Text("\(Int(p * 100))%").font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                        } else if !ready {
                            Label(String(format: "Download %.1f GB", m.sizeGB), systemImage: "arrow.down.circle")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(Color.accentColor)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .stroke(selected && ready ? Color.accentColor : .clear, lineWidth: 2)
                    )
                }
                .buttonStyle(.plain)
                .contextMenu {
                    if ready {
                        Button("Delete model", systemImage: "trash", role: .destructive) { studio.delete(m) }
                    }
                }
            }
        }
    }
}

struct PromptCard: View {
    @EnvironmentObject private var studio: ImageStudio
    @Binding var prompt: String
    var focused: FocusState<Bool>.Binding

    private var ready: Bool { studio.downloaded.contains(studio.model.id) }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            TextField("Describe the image… (English works best)", text: $prompt, axis: .vertical)
                .focused(focused)
                .lineLimit(3...6)
                .padding(12)
                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            DisclosureGroup("Advanced") {
                VStack(alignment: .leading, spacing: 10) {
                    TextField("Negative prompt", text: $studio.negative, axis: .vertical)
                        .font(.footnote)
                        .lineLimit(1...3)
                    Stepper("Steps \(studio.steps)", value: $studio.steps, in: 4...40, step: 2)
                    SecureField("fal.ai API key (cloud models)", text: $studio.falKey)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .font(.footnote)
                    VStack(alignment: .leading) {
                        Text("Guidance \(studio.guidance, specifier: "%.1f")")
                        Slider(value: $studio.guidance, in: 2...12, step: 0.5)
                    }
                }
                .padding(.top, 8)
            }
            .font(.subheadline)
            switch studio.state {
            case .generating(let step, let total):
                ProgressBlock(title: "Step \(step) of \(total)", value: Double(step) / Double(max(total, 1)))
            case .loading:
                ProgressBlock(title: "Preparing the model… the first run takes longer", value: nil)
            case .remote:
                ProgressBlock(title: "Generating on fal.ai…", value: nil)
            default:
                Button {
                    focused.wrappedValue = false
                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    studio.generate(prompt)
                } label: {
                    Label("Generate", systemImage: "wand.and.stars")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(ready && !prompt.isEmpty ? Color.accentColor : Color(.tertiarySystemFill),
                                    in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                        .foregroundStyle(ready && !prompt.isEmpty ? .white : .secondary)
                }
                .buttonStyle(.plain)
                .disabled(!ready || prompt.trimmingCharacters(in: .whitespaces).isEmpty || studio.busy)
            }
        }
    }
}

struct ProgressBlock: View {
    @EnvironmentObject private var studio: ImageStudio
    let title: String
    let value: Double?

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                Text(title).font(.subheadline.weight(.medium)).monospacedDigit()
                if let value {
                    ProgressView(value: value).tint(.accentColor)
                } else {
                    ProgressView().progressViewStyle(.linear)
                }
            }
            Button("Cancel", systemImage: "xmark.circle.fill") { studio.cancel() }
                .labelStyle(.iconOnly)
                .font(.title2)
                .foregroundStyle(.secondary)
        }
        .padding(14)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}
