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
            // Task 6: pair (0,1)'s left window is chunk 0, whose real content
            // is shrunk by leadingPadSamples (zero-pad prepend), so its
            // trusted content ends `leadingPad` earlier than a normal
            // window's nominal `chunkSamples - trailingTrust`. Every later
            // left window is unpadded and keeps the nominal bound.
            let leftTrustEnd =
                s == 0
                ? s + layout.chunkSamples - policy.leadingPadSamples - policy.trailingTrustSamples
                : s + layout.chunkSamples - policy.trailingTrustSamples
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
    ///
    /// Task 6: chunk 0's real-audio content is now shrunk by
    /// `leadingPadSamples` (its head is sacrificed to a zero-pad prepend),
    /// so window 1's own `latestCoveredStart` bound is `leadingPadSamples`
    /// tighter than the naive `1 * stride` target — and because this
    /// policy's stride is calibrated with zero slack (the collapse
    /// property above), that one-time tightening carries forward
    /// unchanged into every later window's bound too (each window's bound
    /// is `previousStart + chunkSamples - minimumOverlapSamples`, and
    /// `previousStart` itself already reflects the shift). Pocket
    /// boundaries are therefore derived from the same recurrence the
    /// aligner itself uses — mirroring `silenceAlignedChunkStarts`'s
    /// `latestCoveredStart` formula including the first-transition extra
    /// pull-back — rather than hardcoded against the unshifted `k * stride`
    /// grid.
    func testGridInvariantWithSilencePockets() throws {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        // 95s of near-silent "speech" — matches the uniform test's duration
        // so the same >= 4 window count guarantee holds.
        var audio = [Float](repeating: 0.02, count: 95 * ASRConstants.sampleRate)
        let layoutProbe =
            ChunkProcessor(audioSamples: audio, edgePolicy: policy)
            .chunkLayoutForTesting(melChunkContext: false, modelVersion: .v3)

        // Carve a ~2-frame silent pocket just before each of the first three
        // reachable bounds, mirroring the carving pattern in
        // ChunkProcessorTests.testNoMelV3ChunkStartsPreferNearbySilence, but
        // computed via the same latestCoveredStart recurrence
        // silenceAlignedChunkStarts uses (see doc comment) so the pockets
        // land inside the aligner's actually-reachable search band.
        var boundaries: [Int] = []
        var cursor = 0
        for k in 1...3 {
            let extraFirstTransitionPullBack = (k == 1) ? policy.leadingPadSamples : 0
            let latestCoveredStart =
                cursor + layoutProbe.chunkSamples - layoutProbe.minimumOverlapSamples - extraFirstTransitionPullBack
            let boundary = latestCoveredStart - 3 * frameSamples
            boundaries.append(boundary)
            cursor = boundary
        }
        for boundary in boundaries {
            for index in (boundary - frameSamples)..<(boundary + frameSamples) {
                audio[index] = 0
            }
        }

        let processor = ChunkProcessor(audioSamples: audio, edgePolicy: policy)
        let starts = try processor.chunkStartsForTesting(melChunkContext: false, modelVersion: .v3)
        let layout = processor.chunkLayoutForTesting(melChunkContext: false, modelVersion: .v3)
        XCTAssertGreaterThanOrEqual(starts.count, 4, "18.88s stride over 95s needs >= 4 more windows")

        for (s, sNext) in zip(starts, starts.dropFirst()) {
            // Task 6: pair (0,1)'s left window is chunk 0 — tighter bound,
            // see the matching comment in testGridInvariantOnUniformSpeech.
            let leftTrustEnd =
                s == 0
                ? s + layout.chunkSamples - policy.leadingPadSamples - policy.trailingTrustSamples
                : s + layout.chunkSamples - policy.trailingTrustSamples
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
        for expectedPocketStart in boundaries {
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

        // Finding 3: positively assert the rescue actually fired — the final
        // start must equal the formula's S_r (frame-ceiled) and the array
        // must have grown past the natural grid's last start.
        let naturalStarts = Array(starts.dropLast())
        let naturalLastStart = naturalStarts.last!
        let trustSpan = layout.chunkSamples - rescuePolicy.trailingTrustSamples
        let rawRescueStart = audio.count - trustSpan
        let expectedRescueStart =
            ((rawRescueStart + frameSamples - 1) / frameSamples) * frameSamples
        XCTAssertEqual(
            lastStart, expectedRescueStart,
            "final start must equal the S_r formula's frame-ceiled rescue start")
        XCTAssertGreaterThan(
            lastStart, naturalLastStart,
            "the rescue entry must actually grow the starts array beyond the natural grid's last start")
    }

    /// Finding 1: the rescue window must actually be dispatched by
    /// `process()`'s loop, not merely appended to `chunkStarts` and then
    /// skipped because the natural last window already satisfies the
    /// coverage-only `isLastChunk` check. Uses the same rescue-triggering
    /// fixture as `testAudioEndFallsInsideFinalWindowTrust`.
    func testRescueWindowIsActuallyDispatched() throws {
        let rescuePolicy = ASREdgePolicy(leadingPadSeconds: 1.0, trailingTrustSeconds: 6.0, matchMarginSeconds: 1.0)
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        let probe = ChunkProcessor(audioSamples: [Float](repeating: 0.02, count: 10), edgePolicy: rescuePolicy)
        let layout = probe.chunkLayoutForTesting(melChunkContext: false, modelVersion: .v3)
        let stride = layout.strideSamples

        let totalLen = 2 * stride - 5 * frameSamples
        var audio = [Float](repeating: 0.02, count: totalLen)
        let pullFrames = 45
        let boundary = stride - pullFrames * frameSamples
        for index in (boundary - frameSamples)..<(boundary + frameSamples) {
            audio[index] = 0
        }

        let processor = ChunkProcessor(audioSamples: audio, edgePolicy: rescuePolicy)
        let starts = try processor.chunkStartsForTesting(melChunkContext: false, modelVersion: .v3)
        XCTAssertGreaterThanOrEqual(starts.count, 2, "fixture must actually trigger a rescue entry")

        let plan = try processor.dispatchPlanForTesting(melChunkContext: false, modelVersion: .v3)

        XCTAssertEqual(
            plan.map(\.start), starts,
            "the dispatched sequence must include every planned start, rescue included")
        XCTAssertEqual(
            plan.last?.start, starts.last,
            "the rescue start must be the last dispatched window")
        XCTAssertEqual(plan.last?.isLastChunk, true, "the rescue window must be flagged as the final chunk")
        XCTAssertEqual(
            plan.dropLast().last?.isLastChunk, false,
            "the natural last window (rescue's predecessor) must no longer be flagged final now that a rescue window follows it")
    }

    /// Legacy path (`edgePolicy == nil`): the dispatched sequence must be
    /// identical to the pre-fix behavior — one window, flagged final as soon
    /// as it covers `totalSamples`.
    func testLegacyDispatchPlanUnchangedWithoutPolicy() throws {
        let audio = [Float](repeating: 0.02, count: 95 * ASRConstants.sampleRate)
        let processor = ChunkProcessor(audioSamples: audio)
        let starts = try processor.chunkStartsForTesting(melChunkContext: false, modelVersion: .v3)
        let plan = try processor.dispatchPlanForTesting(melChunkContext: false, modelVersion: .v3)

        XCTAssertEqual(plan.map(\.start), starts)
        XCTAssertEqual(plan.last?.isLastChunk, true)
        XCTAssertTrue(
            plan.dropLast().allSatisfy { !$0.isLastChunk },
            "only the final dispatched window may be flagged final without an edge policy")
    }

    /// Finding 2: the rescue pair (predecessor window / rescue window) must
    /// also satisfy the grid invariant, exercised by the same
    /// rescue-triggering fixture — the uniform/pocket invariant tests above
    /// never reach the rescue path.
    func testGridInvariantHoldsAcrossRescuePair() throws {
        let rescuePolicy = ASREdgePolicy(leadingPadSeconds: 1.0, trailingTrustSeconds: 6.0, matchMarginSeconds: 1.0)
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        let probe = ChunkProcessor(audioSamples: [Float](repeating: 0.02, count: 10), edgePolicy: rescuePolicy)
        let layout = probe.chunkLayoutForTesting(melChunkContext: false, modelVersion: .v3)
        let stride = layout.strideSamples

        let totalLen = 2 * stride - 5 * frameSamples
        var audio = [Float](repeating: 0.02, count: totalLen)
        let pullFrames = 45
        let boundary = stride - pullFrames * frameSamples
        for index in (boundary - frameSamples)..<(boundary + frameSamples) {
            audio[index] = 0
        }

        let processor = ChunkProcessor(audioSamples: audio, edgePolicy: rescuePolicy)
        let starts = try processor.chunkStartsForTesting(melChunkContext: false, modelVersion: .v3)
        XCTAssertGreaterThanOrEqual(starts.count, 2, "fixture must actually trigger a rescue entry")

        for (s, sNext) in zip(starts, starts.dropFirst()) {
            let leftTrustEnd = s + layout.chunkSamples - rescuePolicy.trailingTrustSamples
            let rightTrustStart = sNext + rescuePolicy.leadingPadSamples
            XCTAssertLessThanOrEqual(
                rightTrustStart + rescuePolicy.matchMarginSamples, leftTrustEnd,
                "windows \(s)/\(sNext) (including the rescue pair) leave an uncovered or margin-less band")
        }
    }

    /// Fix wave 2, Finding 3 follow-up: `regularChunkStarts` (dual-decode
    /// path C) now threads `edgePolicy` and calls the same
    /// `rescueStartIfNeeded` helper `silenceAlignedChunkStarts` (paths A/B)
    /// does. This does NOT make the two grids equal-count in general.
    ///
    /// Task 6 update: `regularChunkStarts` now also applies the
    /// first-transition pull-back (its first stride step is shortened by
    /// `leadingPadSamples`, mirroring `silenceAlignedChunkStarts`'s
    /// first-transition `latestCoveredStart` tightening — chunk 0's real
    /// content is shrunk by the pad regardless of which grid path produced
    /// its start).
    ///
    /// Starved-terminal-window fix update: both grid loops now stop as soon
    /// as the just-appended window's own trust region already reaches
    /// `totalSamples` (`isTerminallyTrustCovered`), instead of blindly
    /// stepping by stride until the *next* candidate start is past
    /// `totalSamples`. For THIS fixture, `regularChunkStarts`' naive
    /// (non-silence-aligned) window 1 already trust-covers the tail after
    /// the first-transition pull-back, so it now stops one window earlier
    /// than before the fix — reopening the 1-window skew against
    /// `silenceAlignedChunkStarts` (whose window 1 is pulled back further by
    /// the carved silence pocket and does need the extra rescue window).
    /// Before this fix, `regularChunkStarts` would have appended that extra
    /// window too, but it would have been payload-starved (audio end only
    /// 9_600 samples past its start) — exactly the bug this fix removes.
    /// The one-window bound `DualDecodeArbitration`'s merge-loop guards
    /// against (`offset + 1 < chosenDecisions.count`) still holds — verified
    /// empirically below.
    func testRegularGridSkewIsAtMostOneWindowVsSilenceAlignedGrid() throws {
        let rescuePolicy = ASREdgePolicy(leadingPadSeconds: 1.0, trailingTrustSeconds: 6.0, matchMarginSeconds: 1.0)
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        let probe = ChunkProcessor(audioSamples: [Float](repeating: 0.02, count: 10), edgePolicy: rescuePolicy)
        let layout = probe.chunkLayoutForTesting(melChunkContext: false, modelVersion: .v3)
        let stride = layout.strideSamples

        let totalLen = 2 * stride - 5 * frameSamples
        var audio = [Float](repeating: 0.02, count: totalLen)
        let pullFrames = 45
        let boundary = stride - pullFrames * frameSamples
        for index in (boundary - frameSamples)..<(boundary + frameSamples) {
            audio[index] = 0
        }

        let processor = ChunkProcessor(audioSamples: audio, edgePolicy: rescuePolicy)
        let silenceAlignedStarts = try processor.silenceAlignedChunkStarts(
            chunkSamples: layout.chunkSamples,
            strideSamples: layout.strideSamples,
            minimumOverlapSamples: layout.minimumOverlapSamples,
            canUseWarmupPrefix: false
        )
        let regularStarts = processor.regularChunkStarts(
            strideSamples: layout.strideSamples,
            chunkSamples: layout.chunkSamples,
            edgePolicy: rescuePolicy
        )

        XCTAssertGreaterThanOrEqual(silenceAlignedStarts.count, 2, "fixture must actually trigger a rescue entry")
        XCTAssertEqual(
            regularStarts.count, 2,
            "regularChunkStarts' naive window 1 already trust-covers the tail for this fixture, so the "
                + "starved-terminal-window fix must stop the grid there rather than append an extra, "
                + "payload-starved window")
        XCTAssertLessThanOrEqual(
            silenceAlignedStarts.count - regularStarts.count, 1,
            "grid skew between the two paths must never exceed the one-window bound the merge loop's bounds guard tolerates")
        XCTAssertLessThanOrEqual(
            audio.count - regularStarts.last!.start, layout.chunkSamples - rescuePolicy.trailingTrustSamples,
            "regularChunkStarts' naive grid never needs a rescue by rescueStartIfNeeded's own trigger condition — it already sits inside the trust span here even with the first-transition pull-back")
    }

    /// Companion positive case: when no silence pull occurs (the aligner
    /// falls back to the naive targets, as in `testGridInvariantOnUniformSpeech`),
    /// `regularChunkStarts` and `silenceAlignedChunkStarts` produce identical
    /// grids, so the shared `rescueStartIfNeeded` call agrees for both
    /// (same `lastStart` in either grid) and counts match exactly.
    func testRegularGridMatchesSilenceAlignedGridWithoutAlignmentPull() throws {
        let policy = ASREdgePolicy.default
        let audio = [Float](repeating: 0.02, count: 95 * ASRConstants.sampleRate)
        let processor = ChunkProcessor(audioSamples: audio, edgePolicy: policy)
        let layout = processor.chunkLayoutForTesting(melChunkContext: false, modelVersion: .v3)

        let silenceAlignedStarts = try processor.silenceAlignedChunkStarts(
            chunkSamples: layout.chunkSamples,
            strideSamples: layout.strideSamples,
            minimumOverlapSamples: layout.minimumOverlapSamples,
            canUseWarmupPrefix: false
        )
        let regularStarts = processor.regularChunkStarts(
            strideSamples: layout.strideSamples,
            chunkSamples: layout.chunkSamples,
            edgePolicy: policy
        )

        XCTAssertEqual(regularStarts.map(\.start), silenceAlignedStarts.map(\.start))
    }

    /// Legacy invariant: `edgePolicy == nil` must still produce a byte-
    /// identical grid from `regularChunkStarts` with the new signature.
    func testRegularChunkStartsUnchangedWithoutPolicy() throws {
        let audio = [Float](repeating: 0.02, count: 95 * ASRConstants.sampleRate)
        let processor = ChunkProcessor(audioSamples: audio)
        let layout = processor.chunkLayoutForTesting(melChunkContext: false, modelVersion: .v3)
        let starts = processor.regularChunkStarts(strideSamples: layout.strideSamples, chunkSamples: layout.chunkSamples)
        XCTAssertEqual(starts.map(\.start), Array(stride(from: 0, to: 95 * ASRConstants.sampleRate, by: layout.strideSamples)))
    }

    /// With edge policy, the first window sacrifices its head to silence pad,
    /// so the SECOND window must start correspondingly earlier.
    func testFirstPairAccountsForLeadingPad() throws {
        let audio = [Float](repeating: 0.02, count: 95 * ASRConstants.sampleRate)
        let processor = ChunkProcessor(audioSamples: audio, edgePolicy: policy)
        let starts = try processor.chunkStartsForTesting(melChunkContext: false, modelVersion: .v3)
        let layout = processor.chunkLayoutForTesting(melChunkContext: false, modelVersion: .v3)
        // Window 0's trusted CONTENT ends at (chunk - pad - trailingTrust) because the
        // pad consumes the head of the model input:
        let win0TrustEnd = layout.chunkSamples - policy.leadingPadSamples - policy.trailingTrustSamples
        XCTAssertLessThanOrEqual(
            starts[1] + policy.leadingPadSamples + policy.matchMarginSamples, win0TrustEnd,
            "second window must compensate for window 0's pad-consumed head")
    }

    /// Starved-terminal-window bug: for a 37.8s clip (604_800 samples) under
    /// `.default`, the naive stride grid places a third window start at
    /// 555_520 with only 49_280 samples (3.08s) of real payload before
    /// clamping to `totalSamples`. `rescueStartIfNeeded`'s guard
    /// (`totalSamples - lastStart > trustSpan`) only fires on *coverage*
    /// shortfall, not on this *payload* shortfall of the terminal window
    /// itself, so no rescue is appended and the merge later drops audio
    /// under this starved window (see ChunkProcessor.swift's
    /// `trustFilteredForMerge`).
    ///
    /// The correct grid never needs the third window at all: window 1
    /// (start 253_440)'s own trust region already reaches past
    /// `totalSamples` (`604_800 - 253_440 = 351_360 <= chunkSamples -
    /// trailingTrust = 382_720`), so no window should ever be appended past
    /// it.
    func testTerminalWindowNotStarvedForClipJustPastGridStart() throws {
        let totalSamples = 604_800
        let audio = [Float](repeating: 0.02, count: totalSamples)
        let processor = ChunkProcessor(audioSamples: audio, edgePolicy: policy)
        let starts = try processor.chunkStartsForTesting(melChunkContext: false, modelVersion: .v3)

        // Explicit no-starved-window property: the terminal window's own
        // payload (totalSamples - lastStart) must never be smaller than
        // what its predecessor's trust region already failed to cover — if
        // the predecessor's trust region already reaches totalSamples, no
        // terminal window may be appended past it at all.
        XCTAssertLessThanOrEqual(
            starts.last!, 253_440,
            "no grid start may sit past 253_440 for this clip: window 1's trust region already "
                + "reaches totalSamples, so any later start is a payload-starved terminal window "
                + "(here: \(totalSamples - starts.last!) samples of real payload)")
        XCTAssertEqual(
            starts, [0, 253_440],
            "window 1's trust region already reaches totalSamples; a third, payload-starved window "
                + "must not be appended")
    }

    /// Companion regression guard: a 32.4s clip (518_400 samples) whose
    /// natural grid is already just [0, 253_440] (terminal payload 16.56s,
    /// nowhere near starved) must be completely unaffected by the fix.
    func testTerminalWindowUnchangedForAlreadyValidClip() throws {
        let totalSamples = 518_400
        let audio = [Float](repeating: 0.02, count: totalSamples)
        let processor = ChunkProcessor(audioSamples: audio, edgePolicy: policy)
        let starts = try processor.chunkStartsForTesting(melChunkContext: false, modelVersion: .v3)

        XCTAssertEqual(starts, [0, 253_440])
    }

    /// Fix wave 1, Finding 1: `silenceAlignedChunkStarts`'s first-transition
    /// `latestCoveredStart` (`previousStart + chunkSamples -
    /// minimumOverlapSamples - firstTransitionExtraOverlap`) has no floor
    /// guard. `ASREdgePolicy.init` is public and unvalidated, so any caller
    /// can construct a policy where `2*leadingPad + trailingTrust +
    /// matchMargin >= chunkSamples` — for such a policy the bound above
    /// goes to/below `previousStart` (0), which would let window 1 start at
    /// or before window 0's start (duplicate window, or a negative sample
    /// offset once consumed downstream). This pathological policy (30s
    /// combined minimum overlap vs. the ~29.92s nominal chunk) triggers
    /// exactly that underflow pre-fix.
    func testFirstTransitionFloorGuardAgainstPathologicalPolicy() throws {
        let pathologicalPolicy = ASREdgePolicy(
            leadingPadSeconds: 15.0, trailingTrustSeconds: 10.0, matchMarginSeconds: 5.0)
        let audio = [Float](repeating: 0.02, count: 5 * ASRConstants.sampleRate)
        let processor = ChunkProcessor(audioSamples: audio, edgePolicy: pathologicalPolicy)
        let starts = try processor.chunkStartsForTesting(melChunkContext: false, modelVersion: .v3)

        XCTAssertGreaterThanOrEqual(starts.count, 2, "fixture must actually reach the first transition")
        XCTAssertGreaterThan(starts[1], 0, "window 1 must never start at or before window 0's start (0)")
        for (s, sNext) in zip(starts, starts.dropFirst()) {
            XCTAssertLessThan(s, sNext, "chunk starts must be strictly increasing")
        }
        for s in starts {
            XCTAssertEqual(s % ASRConstants.samplesPerEncoderFrame, 0, "every start must be frame-aligned")
            XCTAssertGreaterThanOrEqual(s, 0)
            XCTAssertLessThan(s, audio.count, "no start may fall at or past the end of the audio")
        }
    }
}
