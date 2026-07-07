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

    // Synthetic vocab: id 1 = Cyrillic token, id 2 = Latin token.
    private let vocab: [Int: String] = [1: "\u{2581}мак", 2: "\u{2581}mac"]
    private let blankId = 8192

    private func runFilter(
        label: Int, score: Float, threshold: Float?
    ) -> (label: Int, score: Float) {
        var l = label
        var s = score
        TdtDecoderV3.tokenLanguageFilter(
            label: &l, score: &s,
            topKIds: [2, 1], topKLogits: [2.0, 1.0],
            language: .ukrainian, vocabulary: vocab,
            blankId: blankId, confidenceThreshold: threshold
        )
        return (l, s)
    }

    func testNilThresholdSubstitutesWrongScriptToken() {
        // Current behavior preserved: Latin top-1 under a Ukrainian hint
        // is replaced by the best Cyrillic top-K candidate.
        let out = runFilter(label: 2, score: 0.95, threshold: nil)
        XCTAssertEqual(out.label, 1)
    }

    func testHighConfidenceTokenSurvivesWithThreshold() {
        let out = runFilter(label: 2, score: 0.95, threshold: 0.8)
        XCTAssertEqual(out.label, 2, "confident code-switched token must be kept")
        XCTAssertEqual(out.score, 0.95)
    }

    func testLowConfidenceTokenStillSubstituted() {
        let out = runFilter(label: 2, score: 0.5, threshold: 0.8)
        XCTAssertEqual(out.label, 1, "low-confidence wrong-script token must still be filtered")
    }

    func testExactThresholdKeepsToken() {
        // Gate is >= : score equal to threshold bypasses substitution.
        let out = runFilter(label: 2, score: 0.8, threshold: 0.8)
        XCTAssertEqual(out.label, 2)
    }

    func testRightScriptTokenUntouchedEitherWay() {
        let out = runFilter(label: 1, score: 0.3, threshold: 0.8)
        XCTAssertEqual(out.label, 1)
    }

    func testBlankNeverSubstituted() {
        var l = blankId
        var s: Float = 0.1
        TdtDecoderV3.tokenLanguageFilter(
            label: &l, score: &s,
            topKIds: [2, 1], topKLogits: [2.0, 1.0],
            language: .ukrainian, vocabulary: vocab,
            blankId: blankId, confidenceThreshold: 0.8
        )
        XCTAssertEqual(l, blankId)
    }
}
