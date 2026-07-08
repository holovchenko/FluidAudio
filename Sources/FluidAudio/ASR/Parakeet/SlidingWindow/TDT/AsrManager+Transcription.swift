import Foundation

extension AsrManager {

    /// Whether `sampleCount` fits the single-window fast path.
    ///
    /// Without an edge policy this mirrors the legacy threshold: audio up to
    /// `ASRConstants.maxModelSamples` decodes in one shot. With an edge
    /// policy, the fast path is restricted to audio short enough that,
    /// after leading-pad and trailing-trust are reserved, the model's
    /// window still ends in trusted (non-edge) content.
    internal static func usesSingleWindowPath(sampleCount: Int, edgePolicy: ASREdgePolicy?) -> Bool {
        guard let edgePolicy else {
            return sampleCount <= ASRConstants.maxModelSamples
        }
        let threshold =
            ASRConstants.maxModelSamples - edgePolicy.leadingPadSamples - edgePolicy.trailingTrustSamples
        return sampleCount <= threshold
    }

    /// Number of encoder frames spanned by `edgePolicy.leadingPadSamples`, used as the
    /// (negated) `globalFrameOffset` so decoded timestamps land content-relative
    /// despite the leading zero-pad. `nil` policy yields no pad, no offset.
    internal static func leadingPadFrames(edgePolicy: ASREdgePolicy?) -> Int {
        guard let edgePolicy else { return 0 }
        return edgePolicy.leadingPadSamples / ASRConstants.samplesPerEncoderFrame
    }

    /// Drops tokens whose corrected span ends at or before the trusted content start
    /// (`timestamp + duration <= 0`), i.e. tokens the decoder emitted entirely within
    /// the leading zero-pad. Keeps the four lockstep arrays aligned.
    internal static func droppingUntrustedLeadingTokens(
        tokenIds: [Int], timestamps: [Int], confidences: [Float], tokenDurations: [Int]
    ) -> (tokenIds: [Int], timestamps: [Int], confidences: [Float], tokenDurations: [Int]) {
        guard !tokenIds.isEmpty else {
            return (tokenIds, timestamps, confidences, tokenDurations)
        }

        var keptTokenIds: [Int] = []
        var keptTimestamps: [Int] = []
        var keptConfidences: [Float] = []
        var keptDurations: [Int] = []

        for index in 0..<tokenIds.count {
            let timestamp = index < timestamps.count ? timestamps[index] : 0
            let duration = index < tokenDurations.count ? tokenDurations[index] : 0
            guard timestamp + duration > 0 else { continue }

            keptTokenIds.append(tokenIds[index])
            if index < timestamps.count { keptTimestamps.append(timestamp) }
            if index < confidences.count { keptConfidences.append(confidences[index]) }
            if index < tokenDurations.count { keptDurations.append(duration) }
        }

        return (keptTokenIds, keptTimestamps, keptConfidences, keptDurations)
    }

    internal func transcribeWithState(
        _ audioSamples: [Float], decoderState: inout TdtDecoderState, language: Language? = nil
    ) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        let minimumRequiredSamples = ASRConstants.minimumRequiredSamples(forSampleRate: config.sampleRate)
        guard audioSamples.count >= minimumRequiredSamples else { throw ASRError.invalidAudioData }

        let startTime = Date()

        // Route to appropriate processing method based on audio length
        let edgePolicy = config.edgePolicy
        if Self.usesSingleWindowPath(sampleCount: audioSamples.count, edgePolicy: edgePolicy) {
            let leadingPadSamples = edgePolicy?.leadingPadSamples ?? 0
            let paddedInput: [Float] =
                leadingPadSamples > 0
                ? [Float](repeating: 0, count: leadingPadSamples) + audioSamples
                : audioSamples
            let (alignedSamples, frameAlignedLength) = frameAlignedAudio(paddedInput)
            let paddedAudio: [Float] = padAudioIfNeeded(alignedSamples, targetLength: ASRConstants.maxModelSamples)
            let (hypothesis, encoderSequenceLength) = try await executeMLInferenceWithTimings(
                paddedAudio,
                originalLength: frameAlignedLength,
                actualAudioFrames: nil,  // Will be calculated from originalLength
                decoderState: &decoderState,
                isLastChunk: true,  // Single-chunk: always first and last
                globalFrameOffset: -Self.leadingPadFrames(edgePolicy: edgePolicy),
                language: language
            )

            let (tokenIds, timestamps, confidences, tokenDurations) =
                edgePolicy != nil
                ? Self.droppingUntrustedLeadingTokens(
                    tokenIds: hypothesis.ySequence,
                    timestamps: hypothesis.timestamps,
                    confidences: hypothesis.tokenConfidences,
                    tokenDurations: hypothesis.tokenDurations
                )
                : (hypothesis.ySequence, hypothesis.timestamps, hypothesis.tokenConfidences, hypothesis.tokenDurations)

            let result = processTranscriptionResult(
                tokenIds: tokenIds,
                timestamps: timestamps,
                confidences: confidences,
                tokenDurations: tokenDurations,
                encoderSequenceLength: encoderSequenceLength,
                audioSamples: audioSamples,
                processingTime: Date().timeIntervalSince(startTime)
            )

            return result
        }

        // ChunkProcessor handles stateless chunked transcription for long audio
        let processor = ChunkProcessor(audioSamples: audioSamples, edgePolicy: config.edgePolicy)
        let result = try await processor.process(
            using: self,
            startTime: startTime,
            progressHandler: { [weak self] progress in
                guard let self else { return }
                await self.progressEmitter.report(progress: progress)
            },
            language: language
        )

        return result
    }

    /// Chunk transcription that preserves decoder state between calls.
    /// Used by SlidingWindowAsrManager for overlapping-window processing with token deduplication.
    func transcribeChunk(
        _ chunkSamples: [Float],
        decoderState: inout TdtDecoderState,
        previousTokens: [Int] = [],
        isLastChunk: Bool = false,
        language: Language? = nil
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], encoderSequenceLength: Int) {
        let (alignedSamples, frameAlignedLength) = frameAlignedAudio(
            chunkSamples, allowAlignment: previousTokens.isEmpty)
        let padded = padAudioIfNeeded(alignedSamples, targetLength: ASRConstants.maxModelSamples)
        let (hypothesis, encLen) = try await executeMLInferenceWithTimings(
            padded,
            originalLength: frameAlignedLength,
            actualAudioFrames: nil,  // Will be calculated from originalLength
            decoderState: &decoderState,
            contextFrameAdjustment: 0,  // Non-streaming chunks don't use adaptive context
            isLastChunk: isLastChunk,
            language: language
        )

        // Apply token deduplication if previous tokens are provided
        if !previousTokens.isEmpty && hypothesis.hasTokens {
            let (deduped, removedCount) = removeDuplicateTokenSequence(
                previous: previousTokens, current: hypothesis.ySequence)
            let adjustedTimestamps =
                removedCount > 0 ? Array(hypothesis.timestamps.dropFirst(removedCount)) : hypothesis.timestamps
            let adjustedConfidences =
                removedCount > 0
                ? Array(hypothesis.tokenConfidences.dropFirst(removedCount)) : hypothesis.tokenConfidences

            return (deduped, adjustedTimestamps, adjustedConfidences, encLen)
        }

        return (hypothesis.ySequence, hypothesis.timestamps, hypothesis.tokenConfidences, encLen)
    }

    internal func processTranscriptionResult(
        tokenIds: [Int],
        timestamps: [Int] = [],
        confidences: [Float] = [],
        tokenDurations: [Int] = [],
        encoderSequenceLength: Int,
        audioSamples: [Float],
        processingTime: TimeInterval
    ) -> ASRResult {

        let text = convertTokensToText(tokenIds)
        let duration = TimeInterval(audioSamples.count) / TimeInterval(config.sampleRate)

        let resultTimings = createTokenTimings(
            from: tokenIds, timestamps: timestamps, confidences: confidences, tokenDurations: tokenDurations)

        let confidence = calculateConfidence(
            tokenCount: tokenIds.count,
            isEmpty: text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            tokenConfidences: confidences
        )

        return ASRResult(
            text: text,
            confidence: confidence,
            duration: duration,
            processingTime: processingTime,
            tokenTimings: resultTimings
        )
    }

}
