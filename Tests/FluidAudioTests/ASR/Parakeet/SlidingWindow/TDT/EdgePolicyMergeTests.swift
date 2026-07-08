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
