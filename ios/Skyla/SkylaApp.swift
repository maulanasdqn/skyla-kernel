import SwiftUI

@main
struct SkylaApp: App {
    @StateObject private var chat = ChatModel()
    @StateObject private var studio = ImageStudio()

    var body: some Scene {
        WindowGroup {
            TabView {
                Tab("Chat", systemImage: "bubble.left.and.text.bubble.right") {
                    ContentView()
                }
                Tab("Images", systemImage: "photo.on.rectangle.angled") {
                    ImagesView()
                }
            }
            .environmentObject(chat)
            .environmentObject(studio)
            .onAppear { studio.freeMemory = { [weak chat] in chat?.flush() } }
        }
    }
}
