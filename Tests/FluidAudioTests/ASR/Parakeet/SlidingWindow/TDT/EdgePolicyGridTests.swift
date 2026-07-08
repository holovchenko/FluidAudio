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

    /// Same grid invariant as `testGridInvariantOnUniformSpeech`, but with
    /// silence pockets carved near the natural stride targets so the
    /// silence-aligner (see `ChunkProcessor.silenceAlignedChunkStarts`)
    /// actually snaps chunk starts off the naive uniform grid. With
    /// `.default` edgePolicy the trust-region minimum overlap
    /// (leadingPad + trailingTrust + matchMargin = 176_640 samples) makes
    /// `strideSamples` exactly equal to `chunkSamples - minimumOverlapSamples`,
    /// so `latestCoveredStart` collapses onto the naive target itself —
    /// a candidate placed *after* the target is always outside the
    /// aligner's search window and gets silently clamped back to the
    /// uniform grid (verified empirically: it produces the same starts as
    /// the no-pocket case). Pockets are placed slightly *before* each
    /// target instead, which the aligner can and does select.
    func testGridInvariantWithSilencePockets() throws {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        // 95s of near-silent "speech" — matches the uniform test's duration
        // so the same >= 4 window count guarantee holds.
        var audio = [Float](repeating: 0.02, count: 95 * ASRConstants.sampleRate)
        let stride =
            ChunkProcessor(audioSamples: audio, edgePolicy: policy)
            .chunkLayoutForTesting(melChunkContext: false, modelVersion: .v3)
            .strideSamples

        // Carve a ~2-frame silent pocket just before each of the first
        // three policy-stride targets (stride, 2*stride, 3*stride), mirroring
        // the carving pattern in
        // ChunkProcessorTests.testNoMelV3ChunkStartsPreferNearbySilence.
        for k in 1...3 {
            let boundary = k * stride - 3 * frameSamples
            for index in (boundary - frameSamples)..<(boundary + frameSamples) {
                audio[index] = 0
            }
        }

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

        let uniformStarts = try ChunkProcessor(
            audioSamples: [Float](repeating: 0.02, count: 95 * ASRConstants.sampleRate),
            edgePolicy: policy
        ).chunkStartsForTesting(melChunkContext: false, modelVersion: .v3)
        XCTAssertNotEqual(
            starts, uniformStarts,
            "carved pockets must actually pull starts off the uniform grid, or this test degenerates into testGridInvariantOnUniformSpeech")
        for k in 1...3 {
            let expectedPocketStart = k * stride - 3 * frameSamples
            XCTAssertTrue(
                starts.contains(expectedPocketStart),
                "expected a chunk start snapped to the carved pocket at \(expectedPocketStart); starts=\(starts)")
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
