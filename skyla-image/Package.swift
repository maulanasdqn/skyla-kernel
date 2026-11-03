// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "skyla-image",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/apple/ml-stable-diffusion", from: "1.1.0"),
    ],
    targets: [
        .executableTarget(
            name: "skyla-image",
            dependencies: [.product(name: "StableDiffusion", package: "ml-stable-diffusion")]
        ),
    ]
)
