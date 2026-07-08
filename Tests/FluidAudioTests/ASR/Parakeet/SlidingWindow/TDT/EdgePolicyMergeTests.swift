import XCTest

@testable import FluidAudio

final class EdgePolicyMergeTests: XCTestCase {
    private let policy = ASREdgePolicy.default
    private let frame = ASRConstants.samplesPerEncoderFrame

    private func tok(_ id: Int, frameIndex: Int) -> ChunkProcessor.TokenWindow {
        (token: id, timestamp: frameIndex, confidence: 0.9, duration: 1)
    }

    func testLeftTailBeyondTrustPlusMarginIsDropped() {
        // left chunk starts at frame 0, chunk = 374 frames, trailing trust = 75 frames,
        // margin = 25 frames -> keep left tokens starting before frame 374-75+25 = 324.
        let left = [tok(1, frameIndex: 100), tok(2, frameIndex: 320), tok(3, frameIndex: 350)]
        let right = [tok(4, frameIndex: 250), tok(5, frameIndex: 300)]
        let filtered = ChunkProcessor.trustFilteredForMerge(
            left: left, right: right,
            leftChunkStart: 0, rightChunkStart: 236 * frame,
            chunkSamples: 374 * frame, policy: policy, safeIds: [])
        XCTAssertEqual(filtered.left.map(\.token), [1, 2], "frame-350 token is beyond trust+margin")
    }

    func testRightHeadBeforeTrustMinusMarginIsDropped() {
        // right chunk starts at frame 236; leading pad = 38 frames, margin = 25
        // -> keep right tokens ending at/after frame 236+38-25 = 249.
        let left = [tok(1, frameIndex: 100)]
        let right = [tok(4, frameIndex: 240), tok(5, frameIndex: 260)]
        let filtered = ChunkProcessor.trustFilteredForMerge(
            left: left, right: right,
            leftChunkStart: 0, rightChunkStart: 236 * frame,
            chunkSamples: 374 * frame, policy: policy, safeIds: [])
        XCTAssertEqual(filtered.right.map(\.token), [5], "frame-240 token is inside right's damage head")
    }

    func testEffectiveLeftMergeSpanClampsToRealContent() {
        // Left window starts at frame 0; only 350 of the nominal 374 frames
        // of audio actually exist (the natural-last-window-before-rescue
        // case) — the effective span must be the real (smaller) one.
        XCTAssertEqual(
            ChunkProcessor.effectiveLeftMergeSpan(
                nominalChunkSamples: 374 * frame, totalSamples: 350 * frame, leftChunkStart: 0),
            350 * frame)
        // Plenty of audio left: nominal span is unaffected.
        XCTAssertEqual(
            ChunkProcessor.effectiveLeftMergeSpan(
                nominalChunkSamples: 374 * frame, totalSamples: 10_000 * frame, leftChunkStart: 0),
            374 * frame)
    }

    func testLeftTailBeyondClampedSpanTrustIsDroppedEvenWhenNominalSpanWouldKeepIt() {
        // Finding 1: left is the natural last window ahead of a rescue
        // window — its real content span (350 frames) is truncated below
        // the nominal 374. Clamped keep threshold: 0 + 350 - 75 + 25 = 300.
        // Nominal (buggy) keep threshold: 0 + 374 - 75 + 25 = 324.
        // A token at frame 310 sits in (300, 324]: the clamped threshold
        // must drop it even though the nominal one would have kept it.
        let leftChunkStart = 0
        let totalSamples = 350 * frame
        let left = [tok(1, frameIndex: 100), tok(2, frameIndex: 280), tok(3, frameIndex: 310)]
        let right = [tok(4, frameIndex: 250)]

        let clampedSpan = ChunkProcessor.effectiveLeftMergeSpan(
            nominalChunkSamples: 374 * frame, totalSamples: totalSamples, leftChunkStart: leftChunkStart)
        XCTAssertEqual(clampedSpan, 350 * frame, "fixture must actually be truncated below the nominal span")

        let filteredWithClampedSpan = ChunkProcessor.trustFilteredForMerge(
            left: left, right: right,
            leftChunkStart: leftChunkStart, rightChunkStart: 236 * frame,
            chunkSamples: clampedSpan, policy: policy, safeIds: [])
        XCTAssertEqual(
            filteredWithClampedSpan.left.map(\.token), [1, 2],
            "frame-310 token is beyond the clamped trust+margin threshold (300) and must be dropped")

        // Contrast: the pre-fix bug (feeding the nominal chunkSamples
        // constant into a truncated window) would have kept the frame-310
        // token, because 310 < 324.
        let filteredWithNominalSpan = ChunkProcessor.trustFilteredForMerge(
            left: left, right: right,
            leftChunkStart: leftChunkStart, rightChunkStart: 236 * frame,
            chunkSamples: 374 * frame, policy: policy, safeIds: [])
        XCTAssertEqual(
            filteredWithNominalSpan.left.map(\.token), [1, 2, 3],
            "sanity check: the nominal (unclamped) span is the bug this fixture demonstrates")
    }

    func testFilterRightNeverAdmitsLowTrustStraddlingWordViaBackwardSnap() {
        // right chunk starts at frame 236; keep threshold = 236+38-25 = 249.
        // Word straddling the threshold: word-initial piece at frame 242
        // (before the threshold) + its continuation at frame 248 (end 249,
        // straddles it). Finding 2: the not-word-initial branch must search
        // FORWARD for the next word-initial token (frame 260) and drop the
        // whole straddling word — not search backward and re-admit it into
        // right's low-trust head.
        let right = [tok(10, frameIndex: 242), tok(11, frameIndex: 248), tok(12, frameIndex: 260)]
        let filtered = ChunkProcessor.trustFilteredForMerge(
            left: [], right: right,
            leftChunkStart: 0, rightChunkStart: 236 * frame,
            chunkSamples: 374 * frame, policy: policy, safeIds: [10, 12])
        XCTAssertEqual(
            filtered.right.map(\.token), [12],
            "the straddling word (10+11) must be dropped entirely, not re-admitted by a backward snap")
    }

    func testFilterRightFiltersToEmptyWhenNoLaterWordInitialTokenExists() {
        // Same straddling word as above, but no word-initial token exists
        // anywhere after the raw keep index — the right side must filter to
        // empty (left's kept trusted material covers the region) rather
        // than fall back to keeping unfiltered low-trust content.
        let right = [tok(10, frameIndex: 242), tok(11, frameIndex: 248)]
        let filtered = ChunkProcessor.trustFilteredForMerge(
            left: [], right: right,
            leftChunkStart: 0, rightChunkStart: 236 * frame,
            chunkSamples: 374 * frame, policy: policy, safeIds: [10])
        XCTAssertEqual(filtered.right.map(\.token), [], "no later word-initial token exists — must filter to empty")
    }

    func testFilterNeverSplitsAWord() {
        // Word = initial token + continuation (continuation NOT word-initial, i.e.
        // not in safeIds convention used by wordInitialIndex): the cut must snap
        // back so both pieces survive or both drop.
        let left = [tok(1, frameIndex: 100), tok(10, frameIndex: 322), tok(11, frameIndex: 326)]
        // token 10 = word-initial at 322 (< 324 boundary), token 11 = its continuation at 326 (>= 324):
        let filtered = ChunkProcessor.trustFilteredForMerge(
            left: left, right: [],
            leftChunkStart: 0, rightChunkStart: 236 * frame,
            chunkSamples: 374 * frame, policy: policy, safeIds: [10])
        XCTAssertTrue(
            filtered.left.map(\.token) == [1, 10, 11] || filtered.left.map(\.token) == [1],
            "cut must fall on a word boundary — got \(filtered.left.map(\.token))")
    }
}
