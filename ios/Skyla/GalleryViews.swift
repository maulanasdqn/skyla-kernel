import SwiftUI
import UIKit

struct ResultCard: View {
    @EnvironmentObject private var studio: ImageStudio
    let image: GeneratedImage
    let open: () -> Void

    var body: some View {
        VStack(spacing: 10) {
            if let ui = UIImage(contentsOfFile: image.url.path) {
                Image(uiImage: ui)
                    .resizable()
                    .scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .onTapGesture(perform: open)
                HStack {
                    ShareLink(item: Image(uiImage: ui), preview: SharePreview("Skyla image", image: Image(uiImage: ui))) {
                        Label("Share or save", systemImage: "square.and.arrow.up")
                    }
                    Spacer()
                    Button("Delete", systemImage: "trash", role: .destructive) { studio.remove(image) }
                }
                .font(.subheadline)
            }
        }
    }
}

struct GalleryGrid: View {
    @EnvironmentObject private var studio: ImageStudio
    let open: (GeneratedImage) -> Void
    private let columns = Array(repeating: GridItem(.flexible(), spacing: 6), count: 3)

    var body: some View {
        if !studio.gallery.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("Gallery").font(.headline)
                LazyVGrid(columns: columns, spacing: 6) {
                    ForEach(studio.gallery) { item in
                        Button { open(item) } label: {
                            Color.clear
                                .aspectRatio(1, contentMode: .fit)
                                .overlay {
                                    if let ui = UIImage(contentsOfFile: item.url.path) {
                                        Image(uiImage: ui).resizable().scaledToFill()
                                    }
                                }
                                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        }
                        .buttonStyle(.plain)
                        .contextMenu {
                            Button("Delete", systemImage: "trash", role: .destructive) { studio.remove(item) }
                        }
                    }
                }
            }
        }
    }
}

struct ImageViewer: View {
    let image: GeneratedImage
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if let ui = UIImage(contentsOfFile: image.url.path) {
                    Image(uiImage: ui)
                        .resizable()
                        .scaledToFit()
                        .toolbar {
                            ToolbarItem(placement: .topBarLeading) { Button("Done") { dismiss() } }
                            ToolbarItem(placement: .topBarTrailing) {
                                ShareLink(item: Image(uiImage: ui), preview: SharePreview("Skyla image", image: Image(uiImage: ui)))
                            }
                        }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.black)
        }
    }
}
