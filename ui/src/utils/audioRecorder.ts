interface AudioProcessorOptions {
  onProcessingStart?: () => void;
  onProcessingEnd?: () => void;
  onError?: (error: Error) => void;
  onMessage?: (data: {
    transcription: string;
    vad_scores: { arousal: number; dominance: number; valence: number };
  }) => void;
}

export class AudioRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private options: AudioProcessorOptions;

  constructor(options: AudioProcessorOptions = {}) {
    this.options = options;
  }

  async startRecording(): Promise<void> {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => this.processAudio();
      this.mediaRecorder.start();
    } catch (error) {
      this.options.onError?.(error as Error);
    }
  }

  stopRecording(): void {
    if (this.mediaRecorder && this.mediaRecorder.state === "recording") {
      this.mediaRecorder.stop();
      this.mediaRecorder.stream.getTracks().forEach((track) => track.stop());
    }
  }

  private async processAudio(): Promise<void> {
    try {
      this.options.onProcessingStart?.();
      const audioBlob = new Blob(this.audioChunks, { type: "audio/wav" });

      // Ensure proper WAV format by creating a new File with explicit type
      const file = new File([audioBlob], `recording_${Date.now()}.wav`, {
        type: "audio/wav",
      });

      // Create FormData and append the file
      const formData = new FormData();
      formData.append("file", file);

      // Send the file to the backend
      const response = await fetch("http://localhost:8000/process_audio/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.options.onMessage?.(data);
      this.options.onProcessingEnd?.();
    } catch (error) {
      this.options.onError?.(error as Error);
      this.options.onProcessingEnd?.();
    }
  }

  cleanup(): void {
    this.stopRecording();
  }
}
