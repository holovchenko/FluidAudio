import XCTest

@testable import FluidAudio

final class EdgePolicyGridTests: XCTestCase {
    private let policy = ASREdgePolicy.default

    /// Every consecutive window pair must keep a mutual-trust overlap of at
    /// least matchMargin: next window's trusted start begins no later than
    /// the previous window's trusted end minus the margin.
    func testGridInvariantOnUniformSpeech() throws {
        // 95s of constant-energy "speech" — no silence for the aligner to snap to.
        let audio = [Float](repeating: 0.02, count: 95 * ASRConstants.sampleRate)
        let processor = ChunkProcessor(audioSamples: audio, edgePolicy: policy)
        let starts = try processor.chunkStartsForTesting(melChunkContext: false, modelVersion: .v3)
        let layout = processor.chunkLayoutForTesting(melChunkContext: false, modelVersion: .v3)
        XCTAssertGreaterThanOrEqual(starts.count, 4, "18.88s stride over 95s needs >= 4 more windows")
        for (s, sNext) in zip(starts, starts.dropFirst()) {
            let leftTrustEnd = s + layout.chunkSamples - policy.trailingTrustSamples
            let rightTrustStart = sNext + policy.leadingPadSamples
            XCTAssertLessThanOrEqual(
                rightTrustStart + policy.matchMarginSamples, leftTrustEnd,
                "windows \(s)/\(sNext) leave an uncovered or margin-less band")
        }
    }

    func testLegacyGridUnchangedWithoutPolicy() throws {
        let audio = [Float](repeating: 0.02, count: 95 * ASRConstants.sampleRate)
        let legacy = ChunkProcessor(audioSamples: audio)
        let layout = legacy.chunkLayoutForTesting(melChunkContext: false, modelVersion: .v3)
        // 29de8bdf stride: chunk minus 2.0s overlap
        XCTAssertEqual(layout.strideSamples, layout.chunkSamples - 32_000)
    }
}
