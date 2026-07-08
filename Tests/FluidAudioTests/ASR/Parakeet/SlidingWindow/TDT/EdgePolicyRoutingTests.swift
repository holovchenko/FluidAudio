import XCTest

@testable import FluidAudio

final class EdgePolicyRoutingTests: XCTestCase {
    func testLegacyThresholdWithoutPolicy() {
        XCTAssertTrue(AsrManager.usesSingleWindowPath(sampleCount: 480_000, edgePolicy: nil))
        XCTAssertFalse(AsrManager.usesSingleWindowPath(sampleCount: 480_001, edgePolicy: nil))
    }

    func testTrustedThresholdWithPolicy() {
        let p = ASREdgePolicy.default
        XCTAssertTrue(AsrManager.usesSingleWindowPath(sampleCount: 335_360, edgePolicy: p))
        XCTAssertFalse(
            AsrManager.usesSingleWindowPath(sampleCount: 335_361, edgePolicy: p),
            "content past the trusted span must route through ChunkProcessor")
    }

    func testLeadingPadFrames() {
        XCTAssertEqual(AsrManager.leadingPadFrames(edgePolicy: nil), 0)
        XCTAssertEqual(AsrManager.leadingPadFrames(edgePolicy: .default), 38)
    }
}
