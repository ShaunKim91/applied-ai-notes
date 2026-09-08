# Day 3 — Audio AI: Speech Recognition and Synthesis

Audio is continuous and analog; models need discrete numeric input. Today: how a waveform becomes something a transformer can read, and the two directions audio AI runs in.

## From waveform to spectrogram

A raw audio file is just a sequence of amplitude samples over time (e.g. 16,000 numbers per second at 16kHz). That representation buries frequency information inside a long 1D signal that's hard for a model to use directly. The **Short-Time Fourier Transform (STFT)** fixes this by sliding a short window across the waveform and, at each window position, decomposing that slice into its frequency components — producing a 2D **spectrogram**: time on one axis, frequency on the other, intensity as the value.

Human hearing doesn't perceive frequency linearly — we're far more sensitive to differences around 500Hz than around 8000Hz. The **log-Mel spectrogram** rescales the frequency axis onto the Mel scale (matching perceptual sensitivity) and takes the log of the intensities (matching how loudness is perceived), and this is the standard input representation for most speech models, ASR included.

## Two directions

- **ASR / STT (Automatic Speech Recognition / Speech-to-Text)** — audio in, text out.
- **TTS (Text-to-Speech)** — text in, audio out.

They're inverse problems, and mostly use different architectures — an ASR encoder needs to compress a long, noisy signal down to meaning; a TTS model needs to expand compact meaning back out into a long, natural-sounding signal.

## Inside an encoder-decoder ASR model

A model like Whisper processes the log-Mel spectrogram through a transformer **encoder**, producing a sequence of audio representations. A transformer **decoder** then generates the transcript token by token, using **cross-attention** to look back at the encoder's audio representations at every decoding step — deciding, for each word it's about to write, which part of the audio to focus on. This mirrors the text-to-image cross-attention from Week 1 Day 2: same mechanism, different modality bridging text and another signal.

## Running it locally, and a realistic failure

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("standup_notes.wav")
print(result["text"])
```

A common failure mode: domain-specific jargon or product names get transcribed as the nearest common word. A note mentioning "we shipped the Kubernetes ingress patch" might come back as "we shipped the cooper Nettie's ingress patch" — the model has never seen "Kubernetes" often enough relative to phonetically similar filler. A cheap mitigation is passing an **initial prompt** — a short piece of context text that biases decoding toward expected vocabulary:

```python
result = model.transcribe(
    "standup_notes.wav",
    initial_prompt="Kubernetes, ingress, staging, deployment pipeline",
)
```

It doesn't guarantee correctness, but it meaningfully shifts the odds toward the right token when the audio is ambiguous.

## TTS fallback pattern

Not every project needs a neural TTS model. Both major OSes ship a built-in synthesizer accessible without any extra dependency — useful as a zero-setup fallback, or for quick prototyping before reaching for a higher-quality neural voice:

```python
import pyttsx3

engine = pyttsx3.init()
engine.say("Your build finished successfully.")
engine.runAndWait()
```

For higher-quality, more natural-sounding output, swap this for a neural TTS model — but keeping the OS-level fallback means a notification pipeline never has a hard dependency on one being available.

**Takeaway:** speech models operate on log-Mel spectrograms, not raw waveforms; ASR and TTS are inverse problems that mostly need different architectures; and cheap tricks like an initial-prompt hint can meaningfully reduce jargon transcription errors without retraining anything.
