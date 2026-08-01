import Foundation

struct ChunkProcessor {
    let sampleSource: AudioSampleSource
    let totalSamples: Int

    /// Window-edge trust region policy. When set, stride and overlap are
    /// derived from the policy's leading-pad/trailing-trust/match-margin
    /// spans so consecutive windows keep a mutual-trust grid invariant
    /// (see `strideSamples(forChunkSamples:)` / `minimumOverlapSamples`).
    /// `nil` preserves the legacy stride/overlap derivation unchanged.
    let edgePolicy: ASREdgePolicy?

    private let logger = AppLogger(category: "ChunkProcessor")
    typealias TokenWindow = (token: Int, timestamp: Int, confidence: Float, duration: Int)
    private struct TaskResult: Sendable {
        let index: Int
        let tokens: [TokenWindow]
        let workerIndex: Int
    }
    private struct IndexedToken {
        let index: Int
        let token: TokenWindow
        let start: Double
        let end: Double
    }
    struct ChunkStartDecision {
        let start: Int
        let useWarmupPrefix: Bool
    }

    /// One window's dispatch outcome from `process()`'s loop: where it
    /// starts, where it ends (clamped to `totalSamples`), and whether the
    /// pipeline should treat it as the final window (flush/end-of-stream
    /// semantics in `transcribeChunk`).
    struct DispatchedWindow: Equatable {
        let start: Int
        let chunkEnd: Int
        let isLastChunk: Bool
    }

    // Stateless chunking aligned with CoreML reference:
    // - process ~14.96s of audio per window (frame-aligned) to stay under encoder limit
    // - 2.0s overlap (frame-aligned) to give the decoder slack when merging windows
    let overlapSeconds: Double = 2.0

    /// Context samples prepended from previous chunk for mel spectrogram stability (80ms = 1 encoder frame).
    /// The FastConformer encoder's depthwise convolutions need left context for stable output.
    /// Without this, the first frames of a chunk may produce features that cause all-blank predictions.
    ///
    /// Issue #594: on `parakeet-tdt-0.6b-v3-coreml` multilingual long-form
    /// audio this prepend can shift the encoder's first-frame distribution
    /// enough to make the SOS-primed decoder drift to its English-biased prior.
    /// Callers can opt out via `ASRConfig.melChunkContext = false` to
    /// use the v3/no-mel boundary warmup path below.
    private let melContextSamples: Int = ASRConstants.samplesPerEncoderFrame  // 1280 samples = 80ms

    /// Default v3/no-mel path warmup size. v42 intentionally keeps the
    /// non-arbitrated path warmup-free; the opt-in arbitration path's path B
    /// owns the explicit 7-frame warmup probe.
    private let noMelWarmupPrefixFrames: Int = 0

    private var maxModelSamples: Int { ASRConstants.maxModelSamples }

    private var noMelWarmupPrefixSamples: Int {
        noMelWarmupPrefixFrames * ASRConstants.samplesPerEncoderFrame
    }

    /// Effective per-chunk mel-context size based on the runtime flag.
    private func effectiveMelContextSamples(melChunkContext: Bool) -> Int {
        melChunkContext ? melContextSamples : 0
    }

    private func effectiveWarmupPrefixSamples(melChunkContext: Bool, modelVersion: AsrModelVersion?) -> Int {
        guard !melChunkContext, case .v3? = modelVersion else { return 0 }
        return noMelWarmupPrefixSamples
    }

    /// Frame-aligned chunk size that reserves space for the context prepend
    /// (or fills the encoder window when context is disabled).
    private func chunkSamples(melChunkContext: Bool, modelVersion: AsrModelVersion?) -> Int {
        let reserved = effectiveMelContextSamples(melChunkContext: melChunkContext)
        let maxActualChunk = maxModelSamples - reserved
        let raw = max(maxActualChunk - ASRConstants.melHopSize, ASRConstants.samplesPerEncoderFrame)
        return raw / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    /// Legacy (no edge policy) overlap: a flat 2.0s, capped to half the chunk
    /// and frame-aligned.
    private func overlapSamples(forChunkSamples chunkSamples: Int) -> Int {
        let requested = Int(overlapSeconds * Double(ASRConstants.sampleRate))
        let capped = min(requested, chunkSamples / 2)
        return capped / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    /// Minimum trusted overlap the silence-aligned decision loop must
    /// preserve between consecutive windows. With an edge policy this is the
    /// grid invariant's bound (`leadingPad + trailingTrust + matchMargin`);
    /// without one it is the legacy 6-frame minimum.
    private func minimumOverlapSamples(forChunkSamples chunkSamples: Int) -> Int {
        if let edgePolicy {
            return edgePolicy.leadingPadSamples + edgePolicy.trailingTrustSamples + edgePolicy.matchMarginSamples
        }
        return ASRConstants.samplesPerEncoderFrame * 6
    }

    /// Frame-aligned stride between consecutive window starts. With an edge
    /// policy, derived so the grid invariant holds:
    /// `stride = chunkSamples − leadingPad − trailingTrust − matchMargin`.
    /// Without one, legacy stride (`chunkSamples − 2.0s overlap`) is
    /// unchanged.
    private func strideSamples(forChunkSamples chunkSamples: Int) -> Int {
        let raw: Int
        if edgePolicy != nil {
            raw = max(
                chunkSamples - minimumOverlapSamples(forChunkSamples: chunkSamples),
                ASRConstants.samplesPerEncoderFrame
            )
        } else {
            raw = max(chunkSamples - overlapSamples(forChunkSamples: chunkSamples), ASRConstants.samplesPerEncoderFrame)
        }
        return raw / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    /// Frame-aligned stride for the very first window transition (chunk 0 →
    /// chunk 1). With an edge policy, chunk 0's real-audio content is
    /// shrunk by `leadingPadSamples` (it decodes `pad + samples(0..<chunkSamples
    /// − leadingPad)`, see `process()`), so its trusted content ends
    /// `leadingPadSamples` earlier than a normal window's — window 1 must
    /// start that much earlier too, on top of the generic
    /// `minimumOverlapSamples` pull-back already baked into `strideSamples`.
    /// `nil` policy (or a stride already at the 1-frame floor) preserves the
    /// legacy uniform stride unchanged.
    private func firstStrideSamples(strideSamples: Int, edgePolicy: ASREdgePolicy?) -> Int {
        guard let edgePolicy else { return strideSamples }
        return max(
            strideSamples - edgePolicy.leadingPadSamples,
            ASRConstants.samplesPerEncoderFrame
        )
    }

    func chunkLayout(
        melChunkContext: Bool,
        modelVersion: AsrModelVersion?
    ) -> (
        chunkSamples: Int,
        strideSamples: Int,
        melContextSamples: Int,
        warmupPrefixSamples: Int,
        minimumOverlapSamples: Int
    ) {
        let chunkSamples = self.chunkSamples(melChunkContext: melChunkContext, modelVersion: modelVersion)
        let warmupPrefixSamples = effectiveWarmupPrefixSamples(
            melChunkContext: melChunkContext,
            modelVersion: modelVersion
        )
        let stride = strideSamples(forChunkSamples: chunkSamples)
        return (
            chunkSamples: chunkSamples,
            strideSamples: stride,
            melContextSamples: effectiveMelContextSamples(melChunkContext: melChunkContext),
            warmupPrefixSamples: warmupPrefixSamples,
            minimumOverlapSamples: minimumOverlapSamples(forChunkSamples: chunkSamples)
        )
    }

    /// Single source of truth for `process()`'s per-window "is this the last
    /// dispatched window" decision, shared with the DEBUG-only dispatch-plan
    /// seam below so the two never drift.
    ///
    /// Legacy (`edgePolicy == nil`): unchanged — `isLastChunk` is exactly
    /// `candidateEnd >= totalSamples` and `chunkEnd` clamps to `totalSamples`
    /// only on that same window (`min` is a no-op difference from the old
    /// ternary in that case).
    ///
    /// With an edge policy, "last chunk" must be driven by exhaustion of the
    /// planned `chunkStarts` array, not solely by coverage — otherwise a
    /// rescue window appended one-past the natural last window (see
    /// `rescueStartIfNeeded`) is never reached: the natural last window
    /// already satisfies `candidateEnd >= totalSamples`, so a coverage-only
    /// check would break the loop before the rescue window is dispatched.
    private func windowDispatchDecision(
        candidateEnd: Int,
        chunkIndex: Int,
        chunkStartsCount: Int
    ) -> (chunkEnd: Int, isLastChunk: Bool) {
        let coversEnd = candidateEnd >= totalSamples
        let hasMorePlannedStarts = edgePolicy != nil && (chunkIndex + 1) < chunkStartsCount
        let isLastChunk = coversEnd && !hasMorePlannedStarts
        let chunkEnd = min(candidateEnd, totalSamples)
        return (chunkEnd, isLastChunk)
    }

    private func chunkStarts(
        warmupPrefixSamples: Int,
        chunkSamples: Int,
        strideSamples: Int,
        minimumOverlapSamples: Int,
        preferSilenceAlignment: Bool
    ) throws -> [ChunkStartDecision] {
        guard preferSilenceAlignment || warmupPrefixSamples > 0 else {
            return regularChunkStarts(
                strideSamples: strideSamples,
                chunkSamples: chunkSamples,
                edgePolicy: edgePolicy
            )
        }
        return try silenceAlignedChunkStarts(
            chunkSamples: chunkSamples,
            strideSamples: strideSamples,
            minimumOverlapSamples: minimumOverlapSamples,
            canUseWarmupPrefix: warmupPrefixSamples > 0
        )
    }

    /// `chunkSamples` and `edgePolicy` are only consulted to append the
    /// grid-tail rescue window (`rescueStartIfNeeded`) — same rule
    /// `silenceAlignedChunkStarts` applies — so paths A/B/C in the
    /// dual-decode arbitrator produce equal-count grids by construction
    /// whenever an edge policy is set. `edgePolicy == nil` preserves the
    /// legacy grid byte-for-byte (no rescue is ever appended).
    func regularChunkStarts(
        strideSamples: Int,
        chunkSamples: Int,
        edgePolicy: ASREdgePolicy? = nil
    ) -> [ChunkStartDecision] {
        var starts = [ChunkStartDecision(start: 0, useWarmupPrefix: false)]
        // Task 6: chunk 0's real-audio content is shrunk by `leadingPadSamples`
        // (its head is sacrificed to a zero-pad prepend, see `process()`), so
        // window 1 must start correspondingly earlier to keep the grid
        // invariant across the first pair — only the first stride step needs
        // this extra pull-back; every later transition is unaffected (only
        // chunk 0 is physically padded).
        var start = firstStrideSamples(strideSamples: strideSamples, edgePolicy: edgePolicy)
        while start < totalSamples {
            starts.append(ChunkStartDecision(start: start, useWarmupPrefix: false))
            if let edgePolicy, isTerminallyTrustCovered(
                start: start, chunkSamples: chunkSamples, edgePolicy: edgePolicy
            ) {
                break
            }
            start += strideSamples
        }
        if let edgePolicy {
            starts.append(
                contentsOf: rescueStartIfNeeded(
                    lastStart: starts[starts.count - 1].start,
                    chunkSamples: chunkSamples,
                    edgePolicy: edgePolicy
                )
            )
        }
        return starts
    }

    func silenceAlignedChunkStarts(
        chunkSamples: Int,
        strideSamples: Int,
        minimumOverlapSamples: Int = ASRConstants.samplesPerEncoderFrame * 6,
        canUseWarmupPrefix: Bool
    ) throws -> [ChunkStartDecision] {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        let silenceSearchRadiusFrames = max(1, Int((4.0 * Double(ASRConstants.sampleRate)) / Double(frameSamples)))
        let valleySearchRadiusFrames = max(1, Int((0.5 * Double(ASRConstants.sampleRate)) / Double(frameSamples)))
        let halfEnergyWindowSamples = frameSamples

        var starts = [ChunkStartDecision(start: 0, useWarmupPrefix: false)]
        var previousStart = 0
        var target = strideSamples

        while target < totalSamples {
            let targetFrame = target / frameSamples
            // Task 6: the first transition's left window (chunk 0) loses
            // `leadingPadSamples` of real-audio content to the zero-pad
            // prepend in `process()`, so its trusted content ends that much
            // earlier than a normal window's — widen the minimum overlap for
            // this transition only (every later left window is unpadded).
            let firstTransitionExtraOverlap = (previousStart == 0) ? (edgePolicy?.leadingPadSamples ?? 0) : 0
            // Floor guard (review finding 1): for a pathological public-init
            // `ASREdgePolicy` where `2*leadingPad + trailingTrust +
            // matchMargin >= chunkSamples`, the tightened bound above can
            // fall to/below `previousStart`, which would allow window 1 to
            // start at or before window 0's start (duplicate window /
            // negative read offset downstream). Mirror `firstStrideSamples`'s
            // floor: window 1 must start at least one frame after chunk 0.
            let latestCoveredStart = max(
                previousStart + chunkSamples - minimumOverlapSamples - firstTransitionExtraOverlap,
                previousStart + frameSamples
            )
            let targetStart = min(max(targetFrame * frameSamples, previousStart + frameSamples), latestCoveredStart)

            let silenceCandidate = try bestBoundaryCandidate(
                targetFrame: targetFrame,
                searchRadiusFrames: silenceSearchRadiusFrames,
                previousStart: previousStart,
                latestCoveredStart: latestCoveredStart,
                halfEnergyWindowSamples: halfEnergyWindowSamples
            )
            let foundNearSilence = isNearSilenceBoundary(silenceCandidate)

            var bestStart: Int
            var useWarmupPrefix = false
            if foundNearSilence {
                let shouldWarmup =
                    canUseWarmupPrefix ? (try shouldUseWarmupPrefix(at: silenceCandidate.start)) : false
                let compressesSpeechTail: Bool
                if shouldWarmup && silenceCandidate.start < targetStart {
                    compressesSpeechTail = try wouldCompressSpeechTail(
                        candidateStart: silenceCandidate.start,
                        targetStart: targetStart,
                        chunkSamples: chunkSamples,
                        minimumOverlapSamples: minimumOverlapSamples,
                        medianScore: silenceCandidate.medianScore,
                        halfEnergyWindowSamples: halfEnergyWindowSamples
                    )
                } else {
                    compressesSpeechTail = false
                }
                if compressesSpeechTail {
                    bestStart = targetStart
                } else {
                    bestStart = silenceCandidate.start
                    useWarmupPrefix = shouldWarmup
                }
            } else {
                let valleyCandidate = try bestBoundaryCandidate(
                    targetFrame: targetFrame,
                    searchRadiusFrames: valleySearchRadiusFrames,
                    previousStart: previousStart,
                    latestCoveredStart: latestCoveredStart,
                    halfEnergyWindowSamples: halfEnergyWindowSamples
                )
                bestStart = isUsableValleyBoundary(valleyCandidate) ? valleyCandidate.start : targetStart
            }

            if bestStart <= previousStart {
                bestStart = min(previousStart + strideSamples, latestCoveredStart, totalSamples)
            }

            starts.append(
                ChunkStartDecision(
                    start: bestStart,
                    useWarmupPrefix: useWarmupPrefix
                )
            )
            previousStart = bestStart
            if let edgePolicy, isTerminallyTrustCovered(
                start: bestStart, chunkSamples: chunkSamples, edgePolicy: edgePolicy
            ) {
                break
            }
            target += strideSamples
        }

        if let edgePolicy {
            starts.append(
                contentsOf: rescueStartIfNeeded(
                    lastStart: previousStart,
                    chunkSamples: chunkSamples,
                    edgePolicy: edgePolicy
                )
            )
        }

        return starts
    }

    /// Appends one extra window start when the natural grid's last window
    /// leaves audio end outside its trust region (`totalSamples − S_last >
    /// chunkSamples − trailingTrust`). The rescue start is placed so audio
    /// end lands exactly at the trust boundary: `S_r = totalSamples −
    /// (chunkSamples − trailingTrust)`.
    ///
    /// `S_r` is frame-*ceiled* (rounded up), not floored: flooring moves
    /// `S_r` earlier, which *increases* `totalSamples − S_r` and can push it
    /// back past the trust boundary it was placed to satisfy — the opposite
    /// of the intent. Ceiling only shrinks `totalSamples − S_r`, so the
    /// post-rounding coverage property `totalSamples − S_r ≤ chunkSamples −
    /// trailingTrust` still holds (the pre-rounding value satisfies it with
    /// equality).
    ///
    /// The rescue start is also clamped to be later than `lastStart` by at
    /// least one frame — trivially satisfied here since the rescue only
    /// fires when `S_r` (pre-clamp) is already later than `lastStart` (that
    /// is what "the natural grid violates coverage" means arithmetically).
    /// The clamp keeps that invariant explicit rather than assumed.
    ///
    /// A second, upper clamp enforces the grid invariant for the rescue pair
    /// itself: `S_r + leadingPad + matchMargin ≤ S_last + chunkSamples −
    /// trailingTrust`. Without it a policy whose `trailingTrustSeconds` is
    /// large relative to `leadingPad + matchMargin` (`.default` happens to
    /// satisfy the invariant only via the unasserted assumption that the
    /// 4.0s silence-search radius stays under `leadingPad + matchMargin`)
    /// could silently place a rescue start that violates mutual trust with
    /// the previous window. The upper bound is frame-*floored* (rounding
    /// down only shrinks the rescue window's reach, which keeps it inside
    /// the bound rather than pushing it past). If the floored upper bound
    /// falls below the lower bound, no valid placement exists — skip the
    /// rescue rather than emit an invariant-violating start (unreachable for
    /// sane policies; defensive only).
    private func rescueStartIfNeeded(
        lastStart: Int,
        chunkSamples: Int,
        edgePolicy: ASREdgePolicy
    ) -> [ChunkStartDecision] {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        // Safety note (review finding 2): `trustSpan` here is the nominal,
        // un-shrunk span even though chunk 0 may be pad-shrunk by
        // `leadingPadSamples` (Task 6). This is safe specifically for
        // `lastStart == 0`: that only happens when there was a single
        // dispatched window, which per `firstStrideSamples`'s guard requires
        // `firstStrideSamples(...) >= totalSamples` — i.e. chunk 0's grid
        // stride already covers the whole signal. Chunk 0's real (padded)
        // content therefore already reaches `totalSamples`, so the nominal
        // trustSpan can never under-cover and mis-fire a rescue window here.
        let trustSpan = trustSpan(chunkSamples: chunkSamples, edgePolicy: edgePolicy)
        guard totalSamples - lastStart > trustSpan else { return [] }

        let rawRescueStart = totalSamples - trustSpan
        let ceiledRescueStart = ((rawRescueStart + frameSamples - 1) / frameSamples) * frameSamples
        let lowerBound = max(ceiledRescueStart, lastStart + frameSamples)

        let rawUpperBound =
            lastStart + chunkSamples - edgePolicy.trailingTrustSamples - edgePolicy.leadingPadSamples
            - edgePolicy.matchMarginSamples
        let upperBound = floorToFrame(rawUpperBound, frameSamples: frameSamples)

        guard upperBound >= lowerBound else { return [] }

        return [ChunkStartDecision(start: lowerBound, useWarmupPrefix: false)]
    }

    /// Starved-terminal-window guard: true once `start`'s own trust region
    /// (`start ..< start + chunkSamples - trailingTrust`) already reaches
    /// `totalSamples`, meaning no later grid start is needed to cover the
    /// rest of the audio.
    ///
    /// Both grid-generation loops (`regularChunkStarts` /
    /// `silenceAlignedChunkStarts`) previously kept stepping by `stride`
    /// purely because `nextStart < totalSamples`, with no check on whether
    /// the window just appended already trust-covers the tail. For clips
    /// whose duration lands just past a stride-grid start (e.g. a 37.8s
    /// clip under `.default`: window 1 at 253_440 already trust-covers
    /// `totalSamples` = 604_800, since `604_800 − 253_440 = 351_360 ≤
    /// chunkSamples − trailingTrust = 382_720`), the loop still appended one
    /// more window whose real payload before clamping to `totalSamples` was
    /// only a few seconds — a payload-starved terminal window whose tokens
    /// past `rightKeepThreshold` are unusable, and which `rescueStartIfNeeded`
    /// cannot see because its guard only fires on the opposite failure mode
    /// (a *coverage* gap, `totalSamples − lastStart > trustSpan`, not a
    /// *payload* shortfall in an already-appended terminal window).
    ///
    /// Uses the same `trustSpan` formula `rescueStartIfNeeded` uses, so a
    /// grid step and the after-the-fact rescue check share one invariant
    /// rather than each hand-rolling its own threshold.
    private func isTerminallyTrustCovered(
        start: Int,
        chunkSamples: Int,
        edgePolicy: ASREdgePolicy
    ) -> Bool {
        totalSamples - start <= trustSpan(chunkSamples: chunkSamples, edgePolicy: edgePolicy)
    }

    /// Shared invariant: the span of a window's decoded output that is
    /// trusted (not subject to being overwritten/rescued by a later
    /// window), i.e. the chunk minus its trailing untrusted margin. Both
    /// `isTerminallyTrustCovered` (grid-step check) and `rescueStartIfNeeded`
    /// (after-the-fact coverage check) must agree on this span — hand-rolling
    /// it twice risks the two checks silently drifting apart.
    private func trustSpan(chunkSamples: Int, edgePolicy: ASREdgePolicy) -> Int {
        chunkSamples - edgePolicy.trailingTrustSamples
    }

    /// Rounds `value` down to the nearest multiple of `frameSamples`,
    /// correct for negative `value` too (Swift's `/` truncates toward zero,
    /// which rounds a negative value *up* — this rounds it down instead).
    private func floorToFrame(_ value: Int, frameSamples: Int) -> Int {
        let quotient = value / frameSamples
        let remainder = value % frameSamples
        let flooredQuotient = (remainder != 0 && value < 0) ? quotient - 1 : quotient
        return flooredQuotient * frameSamples
    }

    private func bestBoundaryCandidate(
        targetFrame: Int,
        searchRadiusFrames: Int,
        previousStart: Int,
        latestCoveredStart: Int,
        halfEnergyWindowSamples: Int
    ) throws -> (start: Int, score: Float, medianScore: Float) {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        let lowerFrame = max(1, targetFrame - searchRadiusFrames)
        let upperFrame = min((totalSamples - 1) / frameSamples, targetFrame + searchRadiusFrames)
        let targetStart = min(max(targetFrame * frameSamples, previousStart + frameSamples), latestCoveredStart)

        var bestStart = targetStart
        var bestScore = Float.greatestFiniteMagnitude
        var scores: [Float] = []

        if lowerFrame <= upperFrame {
            for frameIndex in lowerFrame...upperFrame {
                let candidate = frameIndex * frameSamples
                if candidate <= previousStart { continue }
                if candidate > latestCoveredStart { continue }
                let score = try boundaryEnergyScore(
                    centeredAt: candidate,
                    halfWindowSamples: halfEnergyWindowSamples
                )
                scores.append(score)
                if score < bestScore {
                    bestScore = score
                    bestStart = candidate
                }
            }
        }

        guard !scores.isEmpty else {
            return (targetStart, Float.greatestFiniteMagnitude, 0)
        }

        let sortedScores = scores.sorted()
        let medianScore = sortedScores[sortedScores.count / 2]
        return (bestStart, bestScore, medianScore)
    }

    private func isNearSilenceBoundary(_ candidate: (start: Int, score: Float, medianScore: Float)) -> Bool {
        candidate.score <= adaptiveBoundaryThreshold(medianScore: candidate.medianScore, ratio: 0.05)
    }

    private func isUsableValleyBoundary(_ candidate: (start: Int, score: Float, medianScore: Float)) -> Bool {
        candidate.score <= adaptiveBoundaryThreshold(medianScore: candidate.medianScore, ratio: 0.35)
    }

    private func adaptiveBoundaryThreshold(medianScore: Float, ratio: Float) -> Float {
        guard medianScore > 0 else { return 0 }
        return medianScore * ratio
    }

    private func wouldCompressSpeechTail(
        candidateStart: Int,
        targetStart: Int,
        chunkSamples: Int,
        minimumOverlapSamples: Int,
        medianScore: Float,
        halfEnergyWindowSamples: Int
    ) throws -> Bool {
        guard medianScore > 0 else { return false }

        let forcedNextBoundary = candidateStart + chunkSamples - minimumOverlapSamples
        guard forcedNextBoundary < totalSamples else { return false }

        let speechLikeThreshold = medianScore * 0.8
        let targetScore = try boundaryEnergyScore(
            centeredAt: targetStart,
            halfWindowSamples: halfEnergyWindowSamples
        )
        let forcedScore = try boundaryEnergyScore(
            centeredAt: forcedNextBoundary,
            halfWindowSamples: halfEnergyWindowSamples
        )
        return targetScore > speechLikeThreshold && forcedScore > speechLikeThreshold
    }

    private func shouldUseWarmupPrefix(at centerSample: Int) throws -> Bool {
        let lookaheadSamples = Int(0.5 * Double(ASRConstants.sampleRate))
        let minimumStableQuietSamples = Int(0.2 * Double(ASRConstants.sampleRate))
        let windowSamples = max(1, ASRConstants.sampleRate / 50)  // 20ms
        let quietRmsThreshold: Float = 0.003

        var offset = 0
        var quietSamples = 0

        while offset < lookaheadSamples {
            let start = centerSample + offset
            guard start < totalSamples else { break }

            let count = min(windowSamples, totalSamples - start, lookaheadSamples - offset)
            guard count > 0 else { break }

            let samples = try readSamples(offset: start, count: count)
            var sum: Float = 0
            for sample in samples {
                sum += sample * sample
            }
            let rms = sqrt(sum / Float(samples.count))
            guard rms < quietRmsThreshold else { break }

            quietSamples += samples.count
            if quietSamples >= minimumStableQuietSamples {
                return false
            }
            offset += samples.count
        }

        return true
    }

    private func boundaryEnergyScore(centeredAt centerSample: Int, halfWindowSamples: Int) throws -> Float {
        let start = max(0, centerSample - halfWindowSamples)
        let end = min(totalSamples, centerSample + halfWindowSamples)
        let count = end - start
        guard count > 0 else { return 0 }

        let samples = try readSamples(offset: start, count: count)
        var sum: Float = 0
        for sample in samples {
            sum += sample * sample
        }
        return sum / Float(count)
    }

    #if DEBUG
    internal func chunkLayoutForTesting(
        melChunkContext: Bool,
        modelVersion: AsrModelVersion?
    ) -> (
        chunkSamples: Int,
        strideSamples: Int,
        melContextSamples: Int,
        warmupPrefixSamples: Int,
        minimumOverlapSamples: Int
    ) {
        chunkLayout(melChunkContext: melChunkContext, modelVersion: modelVersion)
    }

    internal func chunkStartsForTesting(
        melChunkContext: Bool,
        modelVersion: AsrModelVersion?
    ) throws -> [Int] {
        try chunkStartDecisionsForTesting(
            melChunkContext: melChunkContext,
            modelVersion: modelVersion
        ).map(\.start)
    }

    internal func chunkStartDecisionsForTesting(
        melChunkContext: Bool,
        modelVersion: AsrModelVersion?
    ) throws -> [(start: Int, useWarmupPrefix: Bool)] {
        let layout = chunkLayout(melChunkContext: melChunkContext, modelVersion: modelVersion)
        return try chunkStarts(
            warmupPrefixSamples: layout.warmupPrefixSamples,
            chunkSamples: layout.chunkSamples,
            strideSamples: layout.strideSamples,
            minimumOverlapSamples: layout.minimumOverlapSamples,
            preferSilenceAlignment: !melChunkContext && modelVersion == .v3
        ).map { ($0.start, $0.useWarmupPrefix) }
    }

    /// Mirrors `process()`'s window-boundary control flow (start/end/
    /// isLastChunk sequencing) without decoding, so the dispatch-loop fix
    /// for the rescue-window-unreachable bug (Finding 1) can be asserted
    /// directly. Shares `windowDispatchDecision` with `process()` so the
    /// last-chunk decision itself can never drift between the two; only the
    /// loop-stepping mechanics (warmup-sample bookkeeping, chunkStart
    /// advance) are duplicated, matching the existing `chunkStartsForTesting`
    /// convention of a parallel test-only accessor.
    internal func dispatchPlanForTesting(
        melChunkContext: Bool,
        modelVersion: AsrModelVersion?
    ) throws -> [DispatchedWindow] {
        let layout = chunkLayout(melChunkContext: melChunkContext, modelVersion: modelVersion)
        let chunkStarts = try self.chunkStarts(
            warmupPrefixSamples: layout.warmupPrefixSamples,
            chunkSamples: layout.chunkSamples,
            strideSamples: layout.strideSamples,
            minimumOverlapSamples: layout.minimumOverlapSamples,
            preferSilenceAlignment: !melChunkContext && modelVersion == .v3
        )

        var result: [DispatchedWindow] = []
        var chunkDecision = chunkStarts.first ?? ChunkStartDecision(start: 0, useWarmupPrefix: false)
        var chunkStart = chunkDecision.start
        var chunkIndex = 0

        while chunkStart < totalSamples {
            let warmupSamples =
                chunkIndex > 0 && chunkDecision.useWarmupPrefix
                ? min(layout.warmupPrefixSamples, chunkStart) : 0
            let visibleChunkSamples = max(
                ASRConstants.samplesPerEncoderFrame,
                layout.chunkSamples - warmupSamples
            )
            let candidateEnd = chunkStart + visibleChunkSamples
            let (chunkEnd, isLastChunk) = windowDispatchDecision(
                candidateEnd: candidateEnd,
                chunkIndex: chunkIndex,
                chunkStartsCount: chunkStarts.count
            )

            if chunkEnd <= chunkStart { break }

            result.append(DispatchedWindow(start: chunkStart, chunkEnd: chunkEnd, isLastChunk: isLastChunk))

            chunkIndex += 1
            if isLastChunk { break }

            if chunkIndex < chunkStarts.count {
                chunkDecision = chunkStarts[chunkIndex]
                chunkStart = chunkDecision.start
            } else {
                chunkStart += layout.strideSamples
                chunkDecision = ChunkStartDecision(start: chunkStart, useWarmupPrefix: false)
            }
        }

        return result
    }

    internal func mergeTokenWindowsForTesting(
        left: [(token: Int, timestamp: Int, confidence: Float, duration: Int)],
        right: [(token: Int, timestamp: Int, confidence: Float, duration: Int)],
        spliceSafeTokenIds: Set<Int>? = nil,
        caseVariantIds: [Int: Int]? = nil
    ) -> [(token: Int, timestamp: Int, confidence: Float, duration: Int)] {
        mergeChunks(left, right, spliceSafeTokenIds: spliceSafeTokenIds, caseVariantIds: caseVariantIds)
    }
    #endif

    /// Initialize with a streaming audio sample source for memory-efficient processing.
    init(sampleSource: AudioSampleSource, edgePolicy: ASREdgePolicy? = nil) {
        self.sampleSource = sampleSource
        self.totalSamples = sampleSource.sampleCount
        self.edgePolicy = edgePolicy
    }

    /// Convenience initializer for in-memory audio samples.
    init(audioSamples: [Float], edgePolicy: ASREdgePolicy? = nil) {
        self.init(sampleSource: ArrayAudioSampleSource(samples: audioSamples), edgePolicy: edgePolicy)
    }

    func process(
        using manager: AsrManager,
        startTime: Date,
        progressHandler: ((Double) async -> Void)? = nil,
        language: Language? = nil
    ) async throws -> ASRResult {
        let requestedConcurrency = max(1, await manager.parallelChunkConcurrency)
        let workers = await makeWorkerPool(using: manager, count: requestedConcurrency) ?? [manager]
        let decoderLayers = await manager.decoderLayerCount
        let maxModelSamples = self.maxModelSamples
        // Issue #594: opt-out of PR #264's 80ms mel-context prepend. For v3,
        // no-mel uses real-audio warmup plus silence-aligned chunk starts.
        let melChunkContext = await manager.melChunkContext
        let modelVersion = await manager.modelVersion
        let dualDecodeArbitration = await manager.dualDecodeArbitration

        // Dual-decode opt-in (only effective for v3 + no-mel; other paths
        // are not changed by the flag).
        if dualDecodeArbitration, !melChunkContext, modelVersion == .v3 {
            return try await processWithDualDecodeArbitration(
                using: manager,
                workers: workers,
                decoderLayers: decoderLayers,
                maxModelSamples: maxModelSamples,
                modelVersion: modelVersion,
                startTime: startTime,
                progressHandler: progressHandler,
                language: language
            )
        }

        let layout = chunkLayout(melChunkContext: melChunkContext, modelVersion: modelVersion)
        let melContextSamples = layout.melContextSamples
        let warmupPrefixSamples = layout.warmupPrefixSamples
        let chunkSamples = layout.chunkSamples
        let strideSamples = layout.strideSamples
        let chunkStarts = try self.chunkStarts(
            warmupPrefixSamples: warmupPrefixSamples,
            chunkSamples: chunkSamples,
            strideSamples: strideSamples,
            minimumOverlapSamples: layout.minimumOverlapSamples,
            preferSilenceAlignment: !melChunkContext && modelVersion == .v3
        )

        var chunkOutputs: [[TokenWindow]?] = []
        // Nominal (pre-context/warmup) start sample of each dispatched
        // window, parallel to `chunkOutputs` — the same "chunkStart" value
        // the edge-policy grid math (`strideSamples`/`minimumOverlapSamples`
        // above) is derived from. Used to thread real window-edge trust
        // bounds into `mergeChunks`'s trust filter.
        var chunkStartSamples: [Int] = []
        var availableWorkers = Array(workers.indices)
        var inFlight = 0
        var chunkDecision = chunkStarts.first ?? ChunkStartDecision(start: 0, useWarmupPrefix: false)
        var chunkStart = chunkDecision.start
        var chunkIndex = 0

        func collectNextResult(
            _ group: inout ThrowingTaskGroup<TaskResult, Error>
        ) async throws {
            guard inFlight > 0 else { return }
            guard let finished = try await group.next() else { return }
            chunkOutputs[finished.index] = finished.tokens
            availableWorkers.append(finished.workerIndex)
            inFlight -= 1
        }

        try await withThrowingTaskGroup(of: TaskResult.self) { group in
            while chunkStart < totalSamples {
                try Task.checkCancellation()
                let warmupSamples =
                    chunkIndex > 0 && chunkDecision.useWarmupPrefix
                    ? min(warmupPrefixSamples, chunkStart) : 0
                // Task 6: chunk 0 sacrifices `leadingPadSamples` of its head
                // to a silence zero-pad (mirroring the single-window path's
                // treatment of the corpus's decile-0 word-loss edge) — its
                // real-audio content shrinks by that much, matching the
                // first-transition pull-back the grid start functions above
                // already apply.
                let leadingPadSamplesForChunkZero = chunkIndex == 0 ? (edgePolicy?.leadingPadSamples ?? 0) : 0
                let visibleChunkSamples = max(
                    ASRConstants.samplesPerEncoderFrame,
                    chunkSamples - warmupSamples - leadingPadSamplesForChunkZero
                )
                let candidateEnd = chunkStart + visibleChunkSamples
                let (chunkEnd, isLastChunk) = windowDispatchDecision(
                    candidateEnd: candidateEnd,
                    chunkIndex: chunkIndex,
                    chunkStartsCount: chunkStarts.count
                )

                if chunkEnd <= chunkStart {
                    break
                }

                // In the default path, contextSamples means mel/STFT context
                // and is skipped by the decoder. In v3/no-mel mode, the
                // warmup prefix is decoded from frame 0 and only its emitted
                // tokens are suppressed.
                let contextSamples = warmupSamples > 0 ? 0 : (chunkIndex > 0 ? melContextSamples : 0)
                let contextStart = chunkStart - max(warmupSamples, contextSamples)
                let chunkLengthWithContext = chunkEnd - contextStart
                let realAudioSamples = try readSamples(offset: contextStart, count: chunkLengthWithContext)
                // Chunk 0 decodes `[pad(leadingPadSamples zeros) + realAudioSamples]`;
                // `globalFrameOffsetOverride` below shifts decoded timestamps
                // back by the same span so they land content-relative despite
                // the prepended silence.
                let chunkSamplesArray: [Float] =
                    leadingPadSamplesForChunkZero > 0
                    ? [Float](repeating: 0, count: leadingPadSamplesForChunkZero) + realAudioSamples
                    : realAudioSamples
                let emitTokensAfterFrame =
                    warmupSamples > 0 ? chunkStart / ASRConstants.samplesPerEncoderFrame : nil

                if availableWorkers.isEmpty {
                    try await collectNextResult(&group)
                }
                if availableWorkers.isEmpty {
                    availableWorkers.append(0)
                }

                let workerIndex = availableWorkers.removeFirst()
                let worker = workers[workerIndex]
                let index = chunkIndex
                let chunkStartOffset = warmupSamples > 0 ? contextStart : chunkStart
                let globalFrameOffsetOverride: Int? =
                    leadingPadSamplesForChunkZero > 0 ? -AsrManager.leadingPadFrames(edgePolicy: edgePolicy) : nil
                let dropUntrustedLeadingTokens = leadingPadSamplesForChunkZero > 0
                chunkOutputs.append(nil)
                chunkStartSamples.append(chunkStart)

                group.addTask {
                    var decoderState = TdtDecoderState.make(decoderLayers: decoderLayers)
                    decoderState.reset()

                    let (windowTokens, windowTimestamps, windowConfidences, windowDurations) =
                        try await Self
                        .transcribeChunk(
                            samples: chunkSamplesArray,
                            contextSamples: contextSamples,
                            chunkStart: chunkStartOffset,
                            isLastChunk: isLastChunk,
                            using: worker,
                            decoderState: &decoderState,
                            maxModelSamples: maxModelSamples,
                            language: language,
                            emitTokensAfterFrame: emitTokensAfterFrame,
                            initialTimeIndexOverride: emitTokensAfterFrame == nil ? nil : 0,
                            globalFrameOffsetOverride: globalFrameOffsetOverride
                        )

                    guard
                        windowTokens.count == windowTimestamps.count
                            && windowTokens.count == windowConfidences.count
                    else {
                        throw ASRError.processingFailed("Token, timestamp, and confidence arrays are misaligned")
                    }

                    let durations =
                        windowDurations.count == windowTokens.count
                        ? windowDurations : Array(repeating: 0, count: windowTokens.count)

                    let (finalTokens, finalTimestamps, finalConfidences, finalDurations) =
                        dropUntrustedLeadingTokens
                        ? AsrManager.droppingUntrustedLeadingTokens(
                            tokenIds: windowTokens, timestamps: windowTimestamps,
                            confidences: windowConfidences, tokenDurations: durations
                        )
                        : (windowTokens, windowTimestamps, windowConfidences, durations)

                    let windowData: [TokenWindow] = zip(
                        zip(zip(finalTokens, finalTimestamps), finalConfidences), finalDurations
                    ).map {
                        (token: $0.0.0.0, timestamp: $0.0.0.1, confidence: $0.0.1, duration: $0.1)
                    }

                    return TaskResult(index: index, tokens: windowData, workerIndex: workerIndex)
                }
                inFlight += 1
                chunkIndex += 1

                // `chunkEnd < totalSamples` guards against a penultimate
                // window (now possible with an edge policy's rescue window
                // still pending) momentarily reporting 100% progress before
                // the actual last window finishes.
                if let progressHandler, !isLastChunk, chunkEnd < totalSamples {
                    let progress = min(1.0, max(0.0, Double(chunkEnd) / Double(totalSamples)))
                    await progressHandler(progress)
                }

                if isLastChunk {
                    break
                }

                if chunkIndex < chunkStarts.count {
                    chunkDecision = chunkStarts[chunkIndex]
                    chunkStart = chunkDecision.start
                } else {
                    chunkStart += strideSamples
                    chunkDecision = ChunkStartDecision(start: chunkStart, useWarmupPrefix: false)
                }

                if availableWorkers.isEmpty && inFlight > 0 {
                    try await collectNextResult(&group)
                }
            }

            while inFlight > 0 {
                try Task.checkCancellation()
                try await collectNextResult(&group)
            }
        }

        let orderedChunkOutputs = chunkOutputs.compactMap { $0 }

        guard var mergedTokens = orderedChunkOutputs.first else {
            return await manager.processTranscriptionResult(
                tokenIds: [],
                timestamps: [],
                confidences: [],
                encoderSequenceLength: 0,
                audioSamples: [],
                processingTime: Date().timeIntervalSince(startTime)
            )
        }

        if orderedChunkOutputs.count > 1 {
            let vocabulary = await manager.vocabulary
            let spliceSafeTokenIds = Self.spliceSafeTokenIds(vocabulary: vocabulary)
            let caseVariantIds = Self.caseVariantCanonicalIds(vocabulary: vocabulary)
            for (offset, chunk) in orderedChunkOutputs.dropFirst().enumerated() {
                let leftChunkStart = chunkStartSamples[offset]
                // Finding 1: the left window's REAL content span can be
                // truncated below the nominal `chunkSamples` — exactly the
                // rescue scenario (`windowDispatchDecision` clamps
                // `chunkEnd` to `totalSamples`). Pass the clamped span so
                // `trustFilteredForMerge`'s left keep-threshold reflects the
                // window that was actually decoded, not the nominal one.
                mergedTokens = mergeChunks(
                    mergedTokens,
                    chunk,
                    spliceSafeTokenIds: spliceSafeTokenIds,
                    caseVariantIds: caseVariantIds,
                    leftChunkStart: leftChunkStart,
                    rightChunkStart: chunkStartSamples[offset + 1],
                    chunkSamples: Self.effectiveLeftMergeSpan(
                        nominalChunkSamples: layout.chunkSamples,
                        totalSamples: totalSamples,
                        leftChunkStart: leftChunkStart,
                        // Task 6: only the FIRST pair's left window is chunk 0
                        // (the only one physically pad-shrunk); `offset == 0`
                        // here indexes `orderedChunkOutputs.dropFirst()`, so
                        // offset 0 is the (chunk0, chunk1) pair.
                        leadingPadSamples: offset == 0 ? (edgePolicy?.leadingPadSamples ?? 0) : 0
                    )
                )
            }
            // The pairwise merges above already yield tokens in linear (text)
            // order. Do NOT re-sort by timestamp: frame timestamps are coarse
            // (TDT emits several tokens per 80 ms frame, so many are equal) and
            // the two overlapping windows' frame indices don't co-register
            // across a seam, so a timestamp sort reorders same-frame subwords
            // and interleaves subwords from the two windows — the token-order
            // inversion in issue #825 ("Für die" -> "die Für", "im Frühjahr"
            // -> "imüh Frjahr", "Punkt" -> "Pktun"). Preserve the merge order
            // and only clamp timestamps to be non-decreasing so downstream word
            // timing and the seam-gap repair pass stay monotonic.
            mergedTokens = Self.enforceMonotonicTimestamps(mergedTokens)
            mergedTokens = collapseSeamWordDuplicates(mergedTokens, vocabulary: vocabulary)
        } else {
            // Single window: tokens are already emitted in time order; clamp is
            // a no-op but keeps the invariant explicit.
            mergedTokens = Self.enforceMonotonicTimestamps(mergedTokens)
        }

        // Issue #758: the merge above can deterministically drop multi-second
        // spans of clear speech at a chunk seam. Detect suspicious gaps and
        // re-decode each with a fresh window centred on the gap, where the
        // seam does not exist.
        if orderedChunkOutputs.count > 1, mergedTokens.count > 1, await manager.seamGapRepair {
            let vocabulary = await manager.vocabulary
            mergedTokens = try await repairSeamGaps(
                in: mergedTokens,
                using: workers[0],
                decoderLayers: decoderLayers,
                maxModelSamples: maxModelSamples,
                minGapSeconds: await manager.seamGapRepairMinGapSeconds,
                spliceSafeTokenIds: Self.spliceSafeTokenIds(vocabulary: vocabulary),
                vocabulary: vocabulary,
                language: language
            )
        }

        let allTokens = mergedTokens.map { $0.token }
        let allTimestamps = mergedTokens.map { $0.timestamp }
        let allConfidences = mergedTokens.map { $0.confidence }
        let allDurations = mergedTokens.map { $0.duration }

        return await manager.processTranscriptionResult(
            tokenIds: allTokens,
            timestamps: allTimestamps,
            confidences: allConfidences,
            tokenDurations: allDurations,
            encoderSequenceLength: 0,  // Not relevant for chunk processing
            audioSamples: [],
            processingTime: Date().timeIntervalSince(startTime)
        )
    }

    private func makeWorkerPool(using manager: AsrManager, count: Int) async -> [AsrManager]? {
        guard count > 0 else { return nil }
        var workers: [AsrManager] = [manager]
        if count == 1 {
            return workers
        }
        for _ in 1..<count {
            guard let clone = await manager.makeWorkerClone() else {
                return nil
            }
            workers.append(clone)
        }
        logger.debug("ChunkProcessor using worker pool of size \(workers.count)")
        return workers
    }

    func readSamples(offset: Int, count: Int) throws -> [Float] {
        var buffer = [Float](repeating: 0, count: count)
        try buffer.withUnsafeMutableBufferPointer { pointer in
            try sampleSource.copySamples(into: pointer.baseAddress!, offset: offset, count: count)
        }
        return buffer
    }

    static func transcribeChunk(
        samples: [Float],
        contextSamples: Int,
        chunkStart: Int,
        isLastChunk: Bool,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState,
        maxModelSamples: Int,
        language: Language? = nil,
        emitTokensAfterFrame: Int? = nil,
        initialTimeIndexOverride: Int? = nil,
        globalFrameOffsetOverride: Int? = nil
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], durations: [Int]) {
        guard !samples.isEmpty else { return ([], [], [], []) }

        let paddedChunk = manager.padAudioIfNeeded(samples, targetLength: maxModelSamples)

        // Calculate frame count for the ACTUAL audio (excluding prepended context)
        let actualAudioSamples = samples.count - contextSamples
        let actualFrameCount = ASRConstants.calculateEncoderFrames(from: actualAudioSamples)

        // Global frame offset is based on original chunkStart (not context-adjusted start),
        // unless overridden (Task 6: chunk 0's zero-pad prepend needs a negative offset so
        // decoded timestamps land content-relative despite the pad).
        let globalFrameOffset = globalFrameOffsetOverride ?? (chunkStart / ASRConstants.samplesPerEncoderFrame)

        // Context frame adjustment tells decoder to skip the prepended context frames
        let contextFrames = contextSamples / ASRConstants.samplesPerEncoderFrame

        let (hypothesis, encoderSequenceLength) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: samples.count,  // Full length including context
            actualAudioFrames: actualFrameCount,  // Only actual audio frames (excluding context)
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrames,  // Skip context frames in decoder
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset,
            language: language,
            emitTokensAfterGlobalFrame: emitTokensAfterFrame,
            initialTimeIndexOverride: initialTimeIndexOverride
        )

        if hypothesis.isEmpty || encoderSequenceLength == 0 {
            return ([], [], [], [])
        }

        return (hypothesis.ySequence, hypothesis.timestamps, hypothesis.tokenConfidences, hypothesis.tokenDurations)
    }

    /// Token IDs whose vocabulary piece may safely start the portion spliced
    /// in from the `right` window at a seam: SentencePiece word-initial pieces
    /// (`▁` prefix) or punctuation-only pieces (which attach to the previous
    /// word by design). Returns nil for an empty vocabulary so merge behavior
    /// is unchanged when no vocabulary is available (issue #683).
    static func spliceSafeTokenIds(vocabulary: [Int: String]) -> Set<Int>? {
        guard !vocabulary.isEmpty else { return nil }
        var ids = Set<Int>()
        for (id, piece) in vocabulary where isSpliceSafePiece(piece) {
            ids.insert(id)
        }
        return ids
    }

    /// Maps every token ID that has a case-only twin in the vocabulary to a
    /// shared canonical ID, so the overlap matcher can treat e.g. `▁Meeting`
    /// and `▁meeting` as the same word (issue #706).
    ///
    /// A window that begins mid-sentence biases the RNNT decoder to capitalize
    /// its first word as if it started a sentence. In the 2 s overlap the
    /// previous (left) window already heard that word lower-cased with real
    /// left context, but the exact-ID matcher misses the seam pair because the
    /// IDs differ — so the word survives in both windows and decodes twice,
    /// the second copy spuriously capitalized ("the meeting Meeting was").
    /// Folding case at match time lets the seam word anchor and collapse to the
    /// left window's contextually-correct casing.
    ///
    /// Only IDs that actually share a folded piece with another ID are
    /// included, so the map stays small and exact-ID matching is unchanged for
    /// every token without a case twin. Returns nil for an empty vocabulary so
    /// behavior is unchanged when no vocabulary is available.
    static func caseVariantCanonicalIds(vocabulary: [Int: String]) -> [Int: Int]? {
        guard !vocabulary.isEmpty else { return nil }
        var groups: [String: [Int]] = [:]
        for (id, piece) in vocabulary {
            groups[piece.lowercased(), default: []].append(id)
        }
        var canon: [Int: Int] = [:]
        for (folded, ids) in groups where ids.count > 1 {
            // Only groups with a genuine case twin survive; pure-lowercase,
            // punctuation and numeric pieces are unique and stay singletons.
            // Make the all-lower-case variant the canonical ID so a later
            // collapse can tell which copy of a seam duplicate to keep.
            let canonical = ids.first { vocabulary[$0] == folded } ?? ids.min()!
            for id in ids { canon[id] = canonical }
        }
        return canon.isEmpty ? nil : canon
    }

    /// Make token timestamps non-decreasing *without reordering* the stream.
    /// The merged token order is the source of truth for the transcript text
    /// (`convertTokensToText` joins tokens in array order); frame timestamps
    /// are metadata that can be locally out of order across a chunk seam.
    /// Each token that would step backwards in time is clamped up to the
    /// running maximum, so word timing and the seam-gap repair pass see a
    /// monotonic sequence while the text order the merger produced is kept
    /// intact (issue #825).
    static func enforceMonotonicTimestamps(_ tokens: [TokenWindow]) -> [TokenWindow] {
        guard tokens.count > 1 else { return tokens }
        var result = tokens
        var lastTimestamp = result[0].timestamp
        for index in 1..<result.count {
            if result[index].timestamp < lastTimestamp {
                result[index].timestamp = lastTimestamp
            } else {
                lastTimestamp = result[index].timestamp
            }
        }
        return result
    }

    /// Issue #706: drop an adjacent case-only duplicate of a seam *word* left by
    /// a window that re-emitted the seam word as a false sentence start — e.g.
    /// the previous window ended `...we don't have` and the next emitted
    /// `Have a...`, leaving `we don't have Have a`. Works at the word level
    /// (reconstructing SentencePiece words from the token stream) so it catches
    /// multi-token words too — essential for the small Unified subword vocab,
    /// where whole words like `have`/`Have` are several pieces and a token-level
    /// check never sees them as a unit.
    ///
    /// A pair collapses only when the two words are equal up to case, differ in
    /// case, start within the overlap window, and the earlier word does not end
    /// a sentence — so genuine repeats (`that that`), same-case duplicates, and
    /// legitimate sentence boundaries (`...thank you. You said...`) are left
    /// alone. The lower-cased copy is kept (it is the one with real left
    /// context); if neither is lower-case the earlier copy wins.
    func collapseSeamWordDuplicates(
        _ tokens: [TokenWindow],
        vocabulary: [Int: String]
    ) -> [TokenWindow] {
        guard !vocabulary.isEmpty, tokens.count > 1 else { return tokens }
        let overlapFrames = Int((overlapSeconds / ASRConstants.secondsPerEncoderFrame).rounded())

        func piece(_ id: Int) -> String { vocabulary[id] ?? "" }
        func startsWord(_ id: Int) -> Bool {
            let p = piece(id)
            return p.hasPrefix(ASRConstants.sentencePieceWordBoundary) || p.hasPrefix(" ")
        }

        struct Word {
            var tokens: [TokenWindow]
            var core: String
            var startTimestamp: Int
            var endsSentence: Bool
        }

        // Segment the token stream into words on word-initial pieces.
        var words: [Word] = []
        for token in tokens {
            if words.isEmpty || startsWord(token.token) {
                words.append(Word(tokens: [token], core: "", startTimestamp: token.timestamp, endsSentence: false))
            } else {
                words[words.count - 1].tokens.append(token)
            }
        }

        let strippable = CharacterSet.punctuationCharacters.union(.whitespaces)
        for index in words.indices {
            var text = ""
            for token in words[index].tokens {
                text += stripWordBoundaryPrefix(piece(token.token))
            }
            words[index].core = text.trimmingCharacters(in: strippable)
            if let last = text.last { words[index].endsSentence = ".?!:".contains(last) }
        }

        var keep = [Bool](repeating: true, count: words.count)
        var lastKept = -1
        for index in words.indices {
            guard lastKept >= 0 else {
                lastKept = index
                continue
            }
            let previous = words[lastKept]
            let current = words[index]
            let previousCore = previous.core
            let currentCore = current.core

            let isSeamDuplicate =
                !previousCore.isEmpty && !currentCore.isEmpty
                && previousCore != currentCore
                && previousCore.lowercased() == currentCore.lowercased()
                && currentCore.first?.isLetter == true
                && !previous.endsSentence
                && current.startTimestamp - previous.startTimestamp <= overlapFrames

            guard isSeamDuplicate else {
                lastKept = index
                continue
            }

            // Keep the lower-cased copy; if neither is lower-case keep the
            // earlier (left-context) one.
            if currentCore == currentCore.lowercased(), previousCore != previousCore.lowercased() {
                keep[lastKept] = false
                lastKept = index
            } else {
                keep[index] = false
            }
        }

        var result: [TokenWindow] = []
        result.reserveCapacity(tokens.count)
        for index in words.indices where keep[index] {
            result.append(contentsOf: words[index].tokens)
        }
        return result
    }

    /// A piece is splice-safe when decoding it right after another word does
    /// not glue two words together: it either starts a new word (`▁`/space
    /// prefix) or is pure punctuation/symbols.
    static func isSpliceSafePiece(_ piece: String) -> Bool {
        guard !piece.isEmpty else { return false }
        if isWordBoundary(piece) { return true }
        return piece.unicodeScalars.allSatisfy { scalar in
            CharacterSet.punctuationCharacters.contains(scalar)
                || CharacterSet.symbols.contains(scalar)
        }
    }

    /// Finding 1: a merge pair's left window is the nominal `chunkSamples`
    /// wide only when there's enough audio left to fill it. Whenever the
    /// left window is the natural last window ahead of a rescue window
    /// (`windowDispatchDecision` clamps `chunkEnd = min(candidateEnd,
    /// totalSamples)`), its real decoded content span is `totalSamples -
    /// leftChunkStart`, which can be smaller than the nominal constant.
    /// Callers must pass this clamped span — never the nominal one — as
    /// `trustFilteredForMerge`'s `chunkSamples`, or the left keep-threshold
    /// is too permissive and low-trust tail tokens survive the filter.
    ///
    /// Task 6: when the left window is chunk 0 under an edge policy, its
    /// real-audio content is further shrunk by `leadingPadSamples` (the
    /// head sacrificed to the zero-pad prepend, see `process()`) — callers
    /// pass that span via `leadingPadSamples` only for the first merge pair
    /// so `trustFilteredForMerge`'s left keep-threshold reflects chunk 0's
    /// true (pad-shortened) trusted content end, not the nominal
    /// full-width window. Every other left window passes `0` (unaffected).
    internal static func effectiveLeftMergeSpan(
        nominalChunkSamples: Int,
        totalSamples: Int,
        leftChunkStart: Int,
        leadingPadSamples: Int = 0
    ) -> Int {
        min(nominalChunkSamples - leadingPadSamples, totalSamples - leftChunkStart)
    }

    /// Pre-filters an overlapping window pair's tokens down to their
    /// trusted regions before `mergeChunks`'s overlap/matching algorithm
    /// runs. Left tokens whose start frame lies beyond left's trailing
    /// trust boundary plus `matchMargin` are dropped (they're in left's
    /// low-trust tail); right tokens ending before right's leading-pad
    /// boundary minus `matchMargin` are dropped (they're in right's
    /// low-trust head). The margin keeps enough mutual material at each
    /// edge for the existing contiguous/LCS matcher to still find anchors.
    /// Both cuts snap back to a word-initial boundary (via
    /// `wordInitialIndex`) so a word already mid-stream is never split
    /// across the filtered/dropped boundary; when no safe boundary can be
    /// resolved, that side is left unfiltered rather than risk a split.
    ///
    /// `leftChunkStart` / `rightChunkStart` / `chunkSamples` are in
    /// samples; token timestamps are frame indices — this function
    /// converts internally.
    internal static func trustFilteredForMerge(
        left: [TokenWindow],
        right: [TokenWindow],
        leftChunkStart: Int,
        rightChunkStart: Int,
        chunkSamples: Int,
        policy: ASREdgePolicy,
        safeIds: Set<Int>
    ) -> (left: [TokenWindow], right: [TokenWindow]) {
        let frame = ASRConstants.samplesPerEncoderFrame
        let leftChunkStartFrame = leftChunkStart / frame
        let rightChunkStartFrame = rightChunkStart / frame
        let chunkFrames = chunkSamples / frame
        let trailingTrustFrames = policy.trailingTrustSamples / frame
        let leadingPadFrames = policy.leadingPadSamples / frame
        let marginFrames = policy.matchMarginSamples / frame

        let leftKeepThreshold = leftChunkStartFrame + chunkFrames - trailingTrustFrames + marginFrames
        let rightKeepThreshold = rightChunkStartFrame + leadingPadFrames - marginFrames

        let filteredLeft = filterLeftTrailing(left, keepThreshold: leftKeepThreshold, safeIds: safeIds)
        let filteredRight = filterRightLeading(right, keepThreshold: rightKeepThreshold, safeIds: safeIds)
        return (filteredLeft, filteredRight)
    }

    /// Drops left's trailing tokens whose start frame is beyond
    /// `keepThreshold`, snapping the cut back to a word-initial boundary.
    private static func filterLeftTrailing(
        _ tokens: [TokenWindow],
        keepThreshold: Int,
        safeIds: Set<Int>
    ) -> [TokenWindow] {
        guard let rawCutIndex = tokens.firstIndex(where: { $0.timestamp > keepThreshold }) else {
            return tokens
        }
        guard !safeIds.isEmpty else {
            // No word-boundary vocabulary supplied — plain threshold cut.
            return Array(tokens[..<rawCutIndex])
        }
        if safeIds.contains(tokens[rawCutIndex].token) {
            return Array(tokens[..<rawCutIndex])
        }
        guard rawCutIndex > 0 else {
            // The very first token is already a mid-word continuation piece
            // with no earlier word-initial token to snap to — nothing safe
            // to drop, leave unfiltered.
            return tokens
        }
        if let wordStart = wordInitialIndex(in: tokens, endingAt: rawCutIndex - 1, safeIds: safeIds) {
            return Array(tokens[..<wordStart])
        }
        // No resolvable word boundary — don't filter this side.
        return tokens
    }

    /// Drops right's leading tokens whose end frame is before
    /// `keepThreshold`, snapping the cut back to a word-initial boundary.
    ///
    /// The not-word-initial branch searches FORWARD for the next
    /// word-initial token, dropping the straddling word's head — mirroring
    /// `filterLeftTrailing`'s bias (and `mergeByMidpoint`'s precedent) of
    /// dropping the straddling word rather than re-admitting low-trust
    /// material. The grid invariant guarantees the region around this
    /// threshold lies inside left's kept trusted material, so the left side
    /// supplies the word; if no later word-initial token exists, this side
    /// filters to empty rather than searching backward into the low-trust
    /// head.
    private static func filterRightLeading(
        _ tokens: [TokenWindow],
        keepThreshold: Int,
        safeIds: Set<Int>
    ) -> [TokenWindow] {
        guard let rawKeepStart = tokens.firstIndex(where: { $0.timestamp + $0.duration >= keepThreshold }) else {
            return []
        }
        guard rawKeepStart > 0 else { return tokens }
        guard !safeIds.isEmpty else {
            // No word-boundary vocabulary supplied — plain threshold cut.
            return Array(tokens[rawKeepStart...])
        }
        if safeIds.contains(tokens[rawKeepStart].token) {
            return Array(tokens[rawKeepStart...])
        }
        if let wordStart = wordInitialIndex(in: tokens, startingAt: rawKeepStart, safeIds: safeIds) {
            return Array(tokens[wordStart...])
        }
        // No later word-initial token exists — filter to empty; left's
        // kept trusted material covers this region.
        return []
    }

    /// Index of the word-initial (or punctuation) piece starting the word
    /// that contains `anchor`, searching backward, or nil when the stream
    /// begins mid-word. Shared core for the instance-scoped overload below
    /// (used by the seam-splice logic) and `filterLeftTrailing`'s static
    /// context.
    private static func wordInitialIndex(
        in stream: [TokenWindow],
        endingAt anchor: Int,
        safeIds: Set<Int>
    ) -> Int? {
        var index = anchor
        while index >= 0 {
            if safeIds.contains(stream[index].token) { return index }
            index -= 1
        }
        return nil
    }

    /// Index of the next word-initial (or punctuation) piece at or after
    /// `anchor`, searching forward, or nil when no later word-initial token
    /// exists. Used by `filterRightLeading` to drop a straddling word's head
    /// rather than re-admit low-trust material by searching backward.
    private static func wordInitialIndex(
        in stream: [TokenWindow],
        startingAt anchor: Int,
        safeIds: Set<Int>
    ) -> Int? {
        var index = anchor
        while index < stream.count {
            if safeIds.contains(stream[index].token) { return index }
            index += 1
        }
        return nil
    }

    func mergeChunks(
        _ left: [TokenWindow],
        _ right: [TokenWindow],
        spliceSafeTokenIds: Set<Int>? = nil,
        caseVariantIds: [Int: Int]? = nil,
        leftChunkStart: Int? = nil,
        rightChunkStart: Int? = nil,
        chunkSamples: Int? = nil
    ) -> [TokenWindow] {
        var left = left
        var right = right
        if let edgePolicy, let leftChunkStart, let rightChunkStart, let chunkSamples {
            let filtered = Self.trustFilteredForMerge(
                left: left,
                right: right,
                leftChunkStart: leftChunkStart,
                rightChunkStart: rightChunkStart,
                chunkSamples: chunkSamples,
                policy: edgePolicy,
                safeIds: spliceSafeTokenIds ?? []
            )
            left = filtered.left
            right = filtered.right
        }

        if left.isEmpty { return right }
        if right.isEmpty { return left }

        let frameDuration = ASRConstants.secondsPerEncoderFrame
        let overlapDuration = overlapSeconds
        let halfOverlapWindow = overlapDuration / 2

        func startTime(of token: TokenWindow) -> Double {
            Double(token.timestamp) * frameDuration
        }

        func endTime(of token: TokenWindow) -> Double {
            startTime(of: token) + frameDuration
        }

        let leftEndTime = endTime(of: left.last!)
        let rightStartTime = startTime(of: right.first!)

        if leftEndTime <= rightStartTime {
            return left + right
        }

        let overlapLeft: [IndexedToken] = left.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            let end = start + frameDuration
            guard end > rightStartTime - overlapDuration else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: end)
        }

        let overlapRight: [IndexedToken] = right.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            guard start < leftEndTime + overlapDuration else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: start + frameDuration)
        }

        guard overlapLeft.count >= 2 && overlapRight.count >= 2 else {
            logger.debug(
                "seam: midpoint fallback (sparse overlap) overlapL=\(overlapLeft.count) overlapR=\(overlapRight.count) boundary=\(String(format: "%.2f", rightStartTime))s"
            )
            return mergeByMidpoint(
                left: left, right: right, leftEndTime: leftEndTime, rightStartTime: rightStartTime,
                frameDuration: frameDuration, spliceSafeTokenIds: spliceSafeTokenIds)
        }

        let minimumPairs = max(overlapLeft.count / 2, 1)

        // EXTRACTED: Contiguous matching using SequenceMatcher
        let timeTolerantMatcher: (IndexedToken, IndexedToken) -> Bool = { [self] l, r in
            tokensMatch(l, r, tolerance: halfOverlapWindow, caseVariantIds: caseVariantIds)
        }

        let contiguousMatches = SequenceMatcher.findContiguousMatches(
            left: overlapLeft,
            right: overlapRight,
            matcher: timeTolerantMatcher
        )

        // Convert SequenceMatch results to index pairs
        let contiguousPairs = contiguousMatches.map { ($0.leftStartIndex, $0.rightStartIndex) }

        if contiguousPairs.count >= minimumPairs {
            logger.debug(
                "seam: contiguous merge pairs=\(contiguousPairs.count) min=\(minimumPairs) overlapL=\(overlapLeft.count) overlapR=\(overlapRight.count) boundary=\(String(format: "%.2f", rightStartTime))s"
            )
            return mergeUsingMatches(
                matches: contiguousPairs,
                overlapLeft: overlapLeft,
                overlapRight: overlapRight,
                left: left,
                right: right,
                spliceSafeTokenIds: spliceSafeTokenIds
            )
        }

        // EXTRACTED: LCS fallback using SequenceMatcher
        let lcsMatches = SequenceMatcher.findLongestCommonSubsequence(
            left: overlapLeft,
            right: overlapRight,
            matcher: timeTolerantMatcher
        )

        guard !lcsMatches.isEmpty else {
            logger.debug(
                "seam: midpoint fallback (LCS empty) overlapL=\(overlapLeft.count) overlapR=\(overlapRight.count) contiguous=\(contiguousPairs.count) min=\(minimumPairs) boundary=\(String(format: "%.2f", rightStartTime))s"
            )
            return mergeByMidpoint(
                left: left, right: right, leftEndTime: leftEndTime, rightStartTime: rightStartTime,
                frameDuration: frameDuration, spliceSafeTokenIds: spliceSafeTokenIds)
        }

        // Map LCS matches directly to pairs (no consolidation)
        // mergeUsingMatches requires one pair per matched element to function correctly
        let lcsPairs = lcsMatches.map { ($0.leftStartIndex, $0.rightStartIndex) }

        logger.debug(
            "seam: LCS merge pairs=\(lcsPairs.count) overlapL=\(overlapLeft.count) overlapR=\(overlapRight.count) boundary=\(String(format: "%.2f", rightStartTime))s"
        )
        return mergeUsingMatches(
            matches: lcsPairs,
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            left: left,
            right: right,
            spliceSafeTokenIds: spliceSafeTokenIds
        )
    }

    private func tokensMatch(
        _ left: IndexedToken,
        _ right: IndexedToken,
        tolerance: Double,
        caseVariantIds: [Int: Int]? = nil
    ) -> Bool {
        guard tokenIdsMatch(left.token.token, right.token.token, caseVariantIds: caseVariantIds) else {
            return false
        }
        let timeDifference = abs(left.start - right.start)
        return timeDifference < tolerance
    }

    /// Two token IDs match when they are equal, or — issue #706 — when they are
    /// case-only variants of the same vocabulary piece (e.g. `▁Meeting`/
    /// `▁meeting`), so a seam word the right window capitalized as a false
    /// sentence start still anchors against the left window's lower-cased copy.
    private func tokenIdsMatch(_ left: Int, _ right: Int, caseVariantIds: [Int: Int]?) -> Bool {
        if left == right { return true }
        guard let caseVariantIds, let lhs = caseVariantIds[left], let rhs = caseVariantIds[right] else {
            return false
        }
        return lhs == rhs
    }

    private func mergeUsingMatches(
        matches: [(Int, Int)],
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        left: [TokenWindow],
        right: [TokenWindow],
        spliceSafeTokenIds: Set<Int>?
    ) -> [TokenWindow] {
        let leftIndices = matches.map { overlapLeft[$0.0].index }
        let rightIndices = matches.map { overlapRight[$0.1].index }

        var result: [TokenWindow] = []

        if let firstLeft = leftIndices.first, firstLeft > 0 {
            result.append(contentsOf: left[..<firstLeft])
        }

        for idx in 0..<matches.count {
            let leftIndex = leftIndices[idx]
            let rightIndex = rightIndices[idx]

            result.append(left[leftIndex])

            guard idx < matches.count - 1 else { continue }

            let nextLeftIndex = leftIndices[idx + 1]
            let nextRightIndex = rightIndices[idx + 1]

            let gapLeft = nextLeftIndex > leftIndex + 1 ? Array(left[(leftIndex + 1)..<nextLeftIndex]) : []
            let gapRight = nextRightIndex > rightIndex + 1 ? Array(right[(rightIndex + 1)..<nextRightIndex]) : []

            if gapRight.count > gapLeft.count {
                result.append(contentsOf: gapRight)
            } else {
                result.append(contentsOf: gapLeft)
            }
        }

        if let lastRight = rightIndices.last, lastRight + 1 < right.count {
            let tail = right[(lastRight + 1)...]
            if let safeIds = spliceSafeTokenIds,
                let firstTail = tail.first,
                !safeIds.contains(firstTail.token)
            {
                // Issue #683: the splice lands mid-word — right's first
                // post-match piece continues the word containing the matched
                // anchor, so splicing here can decode a left-prefix +
                // right-suffix hybrid or glue two words together. Re-splice
                // at a word boundary so exactly one window segments the
                // seam word.
                if let wordStart = Self.wordInitialIndex(in: right, endingAt: lastRight, safeIds: safeIds),
                    popSeamWord(from: &result, safeIds: safeIds)
                {
                    // The right window heard the seam word from its start —
                    // adopt its segmentation of the whole word. (The left
                    // window's chunk often ends mid-word here, so its view
                    // of the word is the truncated one.)
                    result.append(contentsOf: right[wordStart...])
                } else {
                    // The right window was cut mid-word at its stream start
                    // (no word-initial piece before the anchor): the left
                    // window owns the seam word. Complete it with left's own
                    // continuation pieces and resume right at its next
                    // word-initial piece instead of gluing.
                    if let lastLeft = leftIndices.last {
                        var cursor = lastLeft + 1
                        while cursor < left.count, !safeIds.contains(left[cursor].token) {
                            result.append(left[cursor])
                            cursor += 1
                        }
                    }
                    if let resume = tail.firstIndex(where: { safeIds.contains($0.token) }) {
                        result.append(contentsOf: tail[resume...])
                    } else {
                        // No word-initial piece anywhere in the tail — the
                        // right window simply ended mid-word. Keep its
                        // continuation pieces verbatim rather than silently
                        // dropping real content (a possible glue beats
                        // dropping a word).
                        result.append(contentsOf: tail)
                    }
                }
            } else {
                result.append(contentsOf: tail)
            }
        }

        return result
    }

    /// Remove the trailing seam word (continuation pieces plus its
    /// word-initial piece) from `result` so the right window's segmentation
    /// of the same word can replace it. Returns false — leaving `result`
    /// untouched — when no word-initial piece exists in `result` at all.
    /// Bounded only by the start of `result`, symmetric with the unbounded
    /// backward search `wordInitialIndex` does on the `right` side — a fixed
    /// per-word piece cap would false-negative on long seam words.
    private func popSeamWord(from result: inout [TokenWindow], safeIds: Set<Int>) -> Bool {
        var cursor = result.count - 1
        while cursor >= 0 {
            if safeIds.contains(result[cursor].token) {
                result.removeLast(result.count - cursor)
                return true
            }
            cursor -= 1
        }
        return false
    }

    private func mergeByMidpoint(
        left: [TokenWindow],
        right: [TokenWindow],
        leftEndTime: Double,
        rightStartTime: Double,
        frameDuration: Double,
        spliceSafeTokenIds: Set<Int>?
    ) -> [TokenWindow] {
        let cutoff = (leftEndTime + rightStartTime) / 2
        // Token streams are emitted in timestamp order, so the cutoff filter
        // is equivalent to a prefix/suffix split.
        var leftEnd = left.firstIndex { Double($0.timestamp) * frameDuration >= cutoff } ?? left.count
        var rightStart = right.firstIndex { Double($0.timestamp) * frameDuration >= cutoff } ?? right.count
        if let safeIds = spliceSafeTokenIds {
            // Issue #683: a pure time cutoff can split a word. Extend the
            // left stream until the word it started is complete, and drop
            // orphaned continuation pieces (whose word-initial piece was
            // trimmed away) from the head of the right stream.
            if leftEnd > 0 {
                while leftEnd < left.count, !safeIds.contains(left[leftEnd].token) {
                    leftEnd += 1
                }
            }
            // Scan into a temporary index first: only adopt the advanced
            // cutoff if a splice-safe token was actually found ahead of it.
            // If none exists, the loop would otherwise walk `rightStart` all
            // the way to `right.count`, discarding the entire right window —
            // fall back to the original cutoff-based split instead.
            var scanIndex = rightStart
            while scanIndex < right.count, !safeIds.contains(right[scanIndex].token) {
                scanIndex += 1
            }
            if scanIndex < right.count {
                rightStart = scanIndex
            }
        }
        logger.debug(
            "seam: midpoint cut at \(String(format: "%.2f", cutoff))s dropLeftTail=\(left.count - leftEnd) dropRightHead=\(rightStart) leftTotal=\(left.count) rightTotal=\(right.count)"
        )
        return Array(left[..<leftEnd]) + Array(right[rightStart...])
    }

    // MARK: - Seam-gap repair (issue #758)

    /// Maximum number of gap probes per file — a backstop against
    /// pathological inputs (e.g. hours of intermittent noise). A half-hour
    /// conference recording with applause breaks legitimately probes ~12
    /// gaps, and iteration over residual gaps needs headroom beyond that.
    private var maxSeamGapRepairs: Int { 32 }

    /// Per-frame RMS above this counts as speech-like energy inside a gap.
    /// Comfortably above the library's quiet threshold (0.003 in
    /// `shouldUseWarmupPrefix`) so room tone and hiss do not trigger probes.
    private var seamGapSpeechRmsThreshold: Float { 0.008 }

    /// Minimum cumulative speech-like audio inside a gap before it is
    /// probed. Genuine pauses with a stray cough stay untouched.
    private var seamGapMinSpeechSeconds: Double { 0.5 }

    /// A piece consisting solely of punctuation/symbol scalars (e.g. the
    /// "." in "else." = ▁else + .). Skipped when resolving the word bordering
    /// a gap for edge dedupe.
    static func isPunctuationOnlyPiece(_ id: Int, vocabulary: [Int: String]) -> Bool {
        guard let piece = vocabulary[id], !piece.isEmpty else { return false }
        return piece.unicodeScalars.allSatisfy { scalar in
            CharacterSet.punctuationCharacters.contains(scalar)
                || CharacterSet.symbols.contains(scalar)
        }
    }

    /// The token of the word bordering a gap, walking past punctuation-only
    /// pieces (`step` -1 walks left from the token before the gap, +1 walks
    /// right from the token after it).
    static func wordNeighbor(
        in stream: [TokenWindow],
        from index: Int,
        step: Int,
        vocabulary: [Int: String]
    ) -> TokenWindow {
        var neighborIndex = index
        while neighborIndex + step >= 0,
            neighborIndex + step < stream.count,
            isPunctuationOnlyPiece(stream[neighborIndex].token, vocabulary: vocabulary)
        {
            neighborIndex += step
        }
        return stream[neighborIndex]
    }

    /// Filter a probe window's decoded tokens down to the run that may be
    /// spliced into a gap:
    ///
    /// - only tokens strictly inside the gap (one-frame margins), so words
    ///   the merged stream already has are never duplicated;
    /// - never splice in mid-word — the run starts at a word-initial (or
    ///   punctuation) piece (same rule as the seam merge, #683);
    /// - edge dedupe: the probe can re-hear the word bordering the gap at a
    ///   slightly shifted frame, sometimes with different capitalisation
    ///   ("▁for" vs "▁For") — keep the merged stream's copy, not both. The
    ///   tolerance is deliberately tight (6 frames = 0.48s): genuine
    ///   stutters ("I I") re-heard by the probe sit at or beyond it.
    static func spliceCandidate(
        windowTokens: [Int],
        windowTimestamps: [Int],
        windowConfidences: [Float],
        windowDurations: [Int],
        gapStartFrame: Int,
        gapEndFrame: Int,
        leadNeighbor: TokenWindow,
        tailNeighbor: TokenWindow,
        spliceSafeTokenIds: Set<Int>?,
        vocabulary: [Int: String]
    ) -> [TokenWindow] {
        let edgeToleranceFrames = 6
        func samePiece(_ a: Int, _ b: Int) -> Bool {
            if a == b { return true }
            guard let pieceA = vocabulary[a], let pieceB = vocabulary[b] else { return false }
            return pieceA.lowercased() == pieceB.lowercased()
        }

        var candidate: [TokenWindow] = []
        for tokenIndex in 0..<windowTokens.count {
            let timestamp = windowTimestamps[tokenIndex]
            guard timestamp > gapStartFrame, timestamp < gapEndFrame - 1 else { continue }
            candidate.append(
                (
                    token: windowTokens[tokenIndex],
                    timestamp: timestamp,
                    confidence: windowConfidences[tokenIndex],
                    duration: windowDurations[tokenIndex]
                )
            )
        }

        if let safeIds = spliceSafeTokenIds {
            while let first = candidate.first, !safeIds.contains(first.token) {
                candidate.removeFirst()
            }
        }

        while let first = candidate.first,
            samePiece(first.token, leadNeighbor.token),
            abs(first.timestamp - leadNeighbor.timestamp) <= edgeToleranceFrames
        {
            candidate.removeFirst()
        }
        while let last = candidate.last,
            samePiece(last.token, tailNeighbor.token),
            abs(tailNeighbor.timestamp - last.timestamp) <= edgeToleranceFrames
        {
            candidate.removeLast()
        }
        // Removing an edge token can expose continuation pieces (or orphaned
        // punctuation) at the head — re-trim.
        if let safeIds = spliceSafeTokenIds {
            while let first = candidate.first,
                !safeIds.contains(first.token) || isPunctuationOnlyPiece(first.token, vocabulary: vocabulary)
            {
                candidate.removeFirst()
            }
        }

        return candidate
    }

    #if DEBUG
    internal func speechLikeSecondsForTesting(from startSample: Int, to endSample: Int) throws -> Double {
        try speechLikeSeconds(from: startSample, to: endSample)
    }
    #endif

    /// Detect inter-token gaps that plausibly contain dropped speech and
    /// re-decode each with a single fresh window centred on the gap. Because
    /// the window is decoded from silence-free state with no seam inside it,
    /// it recovers spans the chunk merger dropped (issue #758). Only tokens
    /// strictly inside the gap are spliced in, starting at a word-initial
    /// piece; a gap of genuine silence produces no in-gap tokens and the
    /// stream is returned unchanged.
    private func repairSeamGaps(
        in tokens: [TokenWindow],
        using manager: AsrManager,
        decoderLayers: Int,
        maxModelSamples: Int,
        minGapSeconds: Double,
        spliceSafeTokenIds: Set<Int>?,
        vocabulary: [Int: String],
        language: Language?
    ) async throws -> [TokenWindow] {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        let frameDuration = ASRConstants.secondsPerEncoderFrame
        let minGapFrames = max(2, Int(minGapSeconds / frameDuration))
        // Same frame-aligned usable window size the chunker uses (no context
        // reservation — the repair window is decoded standalone).
        let windowSamples = max(
            frameSamples,
            (maxModelSamples - ASRConstants.melHopSize) / frameSamples * frameSamples
        )

        var working = tokens
        var probes = 0
        var probedGapStarts = Set<Int>()
        // A successful repair can leave a residual gap (e.g. recovered
        // speech, then applause, then more dropped speech): iterate so the
        // residual — whose start has moved — gets its own probe. Gaps that
        // yielded nothing keep the same start and are skipped via the memo.
        for _ in 0..<3 {
            var inserts: [TokenWindow] = []

            for index in 0..<(working.count - 1) {
                guard probes < maxSeamGapRepairs else { break }

                let current = working[index]
                let next = working[index + 1]
                // Conservative end of the current token: its decoded duration
                // when present, else one frame (mirrors mergeChunks).
                let gapStartFrame = current.timestamp + max(1, current.duration)
                let gapEndFrame = next.timestamp
                guard gapEndFrame - gapStartFrame >= minGapFrames else { continue }
                guard !probedGapStarts.contains(gapStartFrame) else { continue }

                let gapStartSample = gapStartFrame * frameSamples
                let gapEndSample = min(gapEndFrame * frameSamples, totalSamples)
                guard gapEndSample > gapStartSample else { continue }

                let speechSeconds = try speechLikeSeconds(from: gapStartSample, to: gapEndSample)
                guard speechSeconds >= seamGapMinSpeechSeconds else { continue }
                probedGapStarts.insert(gapStartFrame)
                probes += 1

                // Probe placement matters: the merger dropped this span because
                // the decoder blanks after low-SNR audio, and a probe window that
                // replays the same pre-gap audio can blank the same way. Start
                // the fresh window AT the gap (the decoder cold-starts directly
                // on the dropped speech, without the noise history); fall back
                // to a gap-centred window for spans the first placement misses.
                let gapCenterSample = (gapStartSample + gapEndSample) / 2
                let placements = [gapStartSample, gapCenterSample - windowSamples / 2]
                var recovered: [TokenWindow] = []

                for placement in placements {
                    var windowStart = max(0, min(placement, totalSamples - windowSamples))
                    windowStart = windowStart / frameSamples * frameSamples
                    let windowEnd = min(windowStart + windowSamples, totalSamples)
                    guard windowEnd > windowStart else { continue }

                    var decoderState = TdtDecoderState.make(decoderLayers: decoderLayers)
                    decoderState.reset()
                    let windowAudio = try readSamples(offset: windowStart, count: windowEnd - windowStart)
                    let (windowTokens, windowTimestamps, windowConfidences, windowDurations) =
                        try await Self.transcribeChunk(
                            samples: windowAudio,
                            contextSamples: 0,
                            chunkStart: windowStart,
                            isLastChunk: windowEnd >= totalSamples,
                            using: manager,
                            decoderState: &decoderState,
                            maxModelSamples: maxModelSamples,
                            language: language
                        )

                    guard windowTokens.count == windowTimestamps.count,
                        windowTokens.count == windowConfidences.count
                    else { continue }
                    let durations =
                        windowDurations.count == windowTokens.count
                        ? windowDurations : Array(repeating: 0, count: windowTokens.count)

                    let candidate = Self.spliceCandidate(
                        windowTokens: windowTokens,
                        windowTimestamps: windowTimestamps,
                        windowConfidences: windowConfidences,
                        windowDurations: durations,
                        gapStartFrame: gapStartFrame,
                        gapEndFrame: gapEndFrame,
                        leadNeighbor: Self.wordNeighbor(in: working, from: index, step: -1, vocabulary: vocabulary),
                        tailNeighbor: Self.wordNeighbor(in: working, from: index + 1, step: 1, vocabulary: vocabulary),
                        spliceSafeTokenIds: spliceSafeTokenIds,
                        vocabulary: vocabulary
                    )

                    if !candidate.isEmpty {
                        recovered = candidate
                        break
                    }
                }

                guard !recovered.isEmpty else { continue }
                logger.info(
                    "Seam-gap repair: recovered \(recovered.count) tokens in "
                        + String(format: "%.2fs", Double(gapStartFrame) * frameDuration) + "–"
                        + String(format: "%.2fs", Double(gapEndFrame) * frameDuration) + " gap"
                )
                inserts.append(contentsOf: recovered)
            }

            guard !inserts.isEmpty else { break }
            working.append(contentsOf: inserts)
            working.sort { $0.timestamp < $1.timestamp }
        }

        return working
    }

    /// Cumulative duration of speech-like audio (per-frame RMS above
    /// `seamGapSpeechRmsThreshold`) between two sample offsets.
    private func speechLikeSeconds(from startSample: Int, to endSample: Int) throws -> Double {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        var speechFrames = 0
        var offset = startSample
        while offset + frameSamples <= endSample {
            let samples = try readSamples(offset: offset, count: frameSamples)
            var sum: Float = 0
            for sample in samples {
                sum += sample * sample
            }
            if sqrt(sum / Float(samples.count)) > seamGapSpeechRmsThreshold {
                speechFrames += 1
            }
            offset += frameSamples
        }
        return Double(speechFrames) * ASRConstants.secondsPerEncoderFrame
    }
}
