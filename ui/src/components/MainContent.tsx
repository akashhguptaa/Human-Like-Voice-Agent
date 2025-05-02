import React, { useRef, useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Mic, Square } from "lucide-react";

interface AudioResponse {
  transcription: string;
  vad_scores: {
    arousal: number;
    dominance: number;
    valence: number;
  };
}

export const MainContent = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // Use webm format for recording as it's better supported
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm",
        audioBitsPerSecond: 128000,
      });

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        try {
          setIsUploading(true);

          // Create audio blob from chunks
          const audioBlob = new Blob(audioChunksRef.current, {
            type: "audio/webm",
          });

          // Convert to WAV format
          const wavBlob = await convertToWav(audioBlob);

          const file = new File([wavBlob], `recording_${Date.now()}.wav`, {
            type: "audio/wav",
          });

          const formData = new FormData();
          formData.append("file", file);

          const response = await fetch("http://localhost:8000/health_check/", {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          console.log("Received response:", data);
        } catch (error) {
          console.error("Error processing audio:", error);
        } finally {
          setIsUploading(false);
        }
      };

      // Request data every second
      mediaRecorder.start(1000);
      setIsRecording(true);
    } catch (error) {
      console.error("Error starting recording:", error);
      setIsRecording(false);
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream
        .getTracks()
        .forEach((track) => track.stop());
      setIsRecording(false);
    }
  }, [isRecording]);

  const handleButtonClick = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  return (
    <div className="flex flex-col items-center z-10 bg-white/20 backdrop-blur-sm p-10 rounded-2xl shadow-lg mt-8">
      <h1 className="text-4xl font-bold text-gray-800 mb-2">MANAS-AI</h1>
      <p className="text-lg text-gray-600 max-w-md text-center">
        Experience conversations with AI that feel remarkably human
      </p>

      <motion.button
        className={`mt-8 px-6 py-3 rounded-full text-white font-medium shadow-lg flex items-center gap-2 ${
          isRecording
            ? "bg-red-500 hover:bg-red-600"
            : "bg-gradient-to-r from-blue-400 via-purple-400 to-green-400"
        }`}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={handleButtonClick}
        disabled={isUploading}
      >
        {isUploading ? (
          "Processing..."
        ) : isRecording ? (
          <>
            <Square className="w-5 h-5" />
            Stop Recording
          </>
        ) : (
          <>
            <Mic className="w-5 h-5" />
            Start Conversation
          </>
        )}
      </motion.button>
    </div>
  );
};

// Helper function to convert audio to WAV format
async function convertToWav(audioBlob: Blob): Promise<Blob> {
  // Create an audio context
  const audioContext = new AudioContext();

  // Read the blob as ArrayBuffer
  const arrayBuffer = await audioBlob.arrayBuffer();

  // Decode the audio data
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  // Create a new buffer for the WAV file
  const wavBuffer = audioContext.createBuffer(
    audioBuffer.numberOfChannels,
    audioBuffer.length,
    audioBuffer.sampleRate
  );

  // Copy the audio data
  for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
    wavBuffer.copyToChannel(audioBuffer.getChannelData(channel), channel);
  }

  // Convert to WAV format
  const wavBlob = await audioBufferToWav(wavBuffer);
  return wavBlob;
}

// Helper function to convert AudioBuffer to WAV format
async function audioBufferToWav(buffer: AudioBuffer): Promise<Blob> {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;

  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;

  const dataLength = buffer.length * numChannels * bytesPerSample;
  const bufferLength = 44 + dataLength;

  const arrayBuffer = new ArrayBuffer(bufferLength);
  const view = new DataView(arrayBuffer);

  // Write WAV header
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(view, 36, "data");
  view.setUint32(40, dataLength, true);

  // Write audio data
  const offset = 44;
  const channelData = [];
  for (let i = 0; i < numChannels; i++) {
    channelData.push(buffer.getChannelData(i));
  }

  let pos = 0;
  while (pos < buffer.length) {
    for (let i = 0; i < numChannels; i++) {
      const sample = Math.max(-1, Math.min(1, channelData[i][pos]));
      const value = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      view.setInt16(
        offset + pos * blockAlign + i * bytesPerSample,
        value,
        true
      );
    }
    pos++;
  }

  return new Blob([arrayBuffer], { type: "audio/wav" });
}

// Helper function to write strings to DataView
function writeString(view: DataView, offset: number, string: string): void {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}
