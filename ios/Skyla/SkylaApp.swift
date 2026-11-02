import SwiftUI

@main
struct SkylaApp: App {
    @StateObject private var chat = ChatModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(chat)
        }
    }
}
