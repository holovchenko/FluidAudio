import XCTest
@testable import FluidAudio

final class TokenFilterConfidenceGateTests: XCTestCase {

    func testConfigDefaultsToNilThreshold() {
        XCTAssertNil(ASRConfig.default.tokenFilterConfidenceThreshold)
        XCTAssertNil(ASRConfig().tokenFilterConfidenceThreshold)
    }

    func testConfigStoresExplicitThreshold() {
        let config = ASRConfig(tokenFilterConfidenceThreshold: 0.8)
        XCTAssertEqual(config.tokenFilterConfidenceThreshold, 0.8)
    }
}
