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

    /// Audio whose end would land in the last window's damage tail must get
    /// a rescue window: end-of-audio inside the final window's trust region.
    ///
    /// In-test adaptation (per plan note): neither the brief's hardcoded
    /// 48.5s nor the brief's length-derivation fallback (`starts[1] +
    /// chunkSamples - trailingTrust + 5 frames`) actually violates the
    /// invariant against `.default`'s real silence-aligned grid. This is
    /// provable, not just empirical: the silence aligner can pull a window
    /// start earlier than its naive stride target by at most the hardcoded
    /// search radius (4.0s = 64_000 samples,
    /// `silenceSearchRadiusFrames` in `ChunkProcessor`), so the worst-case
    /// natural gap to audio end is `stride + 64_000` samples — and for
    /// `.default` (leadingPad 3.04s + matchMargin 2.0s = 5.04s > 4.0s
    /// radius), `stride + 64_000 < chunkSamples - trailingTrust` always, so
    /// `.default` can never violate the invariant via this loop regardless
    /// of audio length or silence placement.
    /// A violation requires `radius > leadingPad + matchMargin` (in
    /// samples), which `.default` doesn't satisfy. This test therefore uses
    /// a local policy with a smaller `leadingPad + matchMargin` (2.0s) than
    /// the 4.0s search radius, then carves a silence pocket ahead of the
    /// natural stride target — well within the documented, reproducible
    /// mechanics above — to pull the window start early enough to violate.
    func testAudioEndFallsInsideFinalWindowTrust() throws {
        let rescuePolicy = ASREdgePolicy(leadingPadSeconds: 1.0, trailingTrustSeconds: 6.0, matchMarginSeconds: 1.0)
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        let probe = ChunkProcessor(audioSamples: [Float](repeating: 0.02, count: 10), edgePolicy: rescuePolicy)
        let layout = probe.chunkLayoutForTesting(melChunkContext: false, modelVersion: .v3)
        let stride = layout.strideSamples

        // Total length just under 2*stride so the natural grid is [0, ~stride]
        // only (no third window). Carve a silence pocket 45 frames (57_600
        // samples) before the stride target — inside the 50-frame search
        // radius — so the aligner snaps the second window's start there
        // instead of the naive target, pulling it back further than
        // leadingPad + matchMargin (32_000 samples) can absorb.
        let totalLen = 2 * stride - 5 * frameSamples
        var audio = [Float](repeating: 0.02, count: totalLen)
        let pullFrames = 45
        let boundary = stride - pullFrames * frameSamples
        for index in (boundary - frameSamples)..<(boundary + frameSamples) {
            audio[index] = 0
        }

        let processor = ChunkProcessor(audioSamples: audio, edgePolicy: rescuePolicy)
        let starts = try processor.chunkStartsForTesting(melChunkContext: false, modelVersion: .v3)
        let lastStart = starts.last!
        XCTAssertLessThanOrEqual(
            audio.count - lastStart, layout.chunkSamples - rescuePolicy.trailingTrustSamples,
            "audio end must sit inside the final window's trust region")
        XCTAssertEqual(lastStart % ASRConstants.samplesPerEncoderFrame, 0)
    }
}
