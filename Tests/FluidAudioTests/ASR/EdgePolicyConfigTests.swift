import XCTest

@testable import FluidAudio

final class EdgePolicyConfigTests: XCTestCase {
    func testDefaultConfigHasNilEdgePolicy() {
        XCTAssertNil(ASRConfig().edgePolicy)
    }

    func testDefaultPolicyValuesAndFrameAlignment() {
        let p = ASREdgePolicy.default
        XCTAssertEqual(p.leadingPadSeconds, 3.04, accuracy: 1e-9)
        XCTAssertEqual(p.trailingTrustSeconds, 6.0, accuracy: 1e-9)
        XCTAssertEqual(p.matchMarginSeconds, 2.0, accuracy: 1e-9)
        for samples in [p.leadingPadSamples, p.trailingTrustSamples, p.matchMarginSamples] {
            XCTAssertEqual(
                samples % ASRConstants.samplesPerEncoderFrame, 0,
                "edge-policy spans must be whole encoder frames")
        }
        XCTAssertEqual(p.leadingPadSamples, 48_640)
        XCTAssertEqual(p.trailingTrustSamples, 96_000)
        XCTAssertEqual(p.matchMarginSamples, 32_000)
    }
}
