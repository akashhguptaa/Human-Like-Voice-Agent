import React, { useRef, useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Mic, Square, Play, Pause } from "lucide-react";
import { useRouter } from "next/navigation";

interface AudioResponse {
  transcription: string;
  vad_scores: {
    arousal: number;
    dominance: number;
    valence: number;
  };
  emotional_response: {
    response: {
      [key: string]: {
        text: string;
        vad: [number, number, number];
      };
    };
  };
  tts_file: string;
}

export const MainContent = () => {
  const router = useRouter();
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [transcription, setTranscription] = useState<string>("");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

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

          const audioBlob = new Blob(audioChunksRef.current, {
            type: "audio/webm",
          });

          const wavBlob = await convertToWav(audioBlob);

          const file = new File([wavBlob], `recording_${Date.now()}.wav`, {
            type: "audio/wav",
          });

          const formData = new FormData();
          formData.append("file", file);

          const response = await fetch("http://localhost:8000/process_audio/", {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const data = await response.json();

          console.log(data);
          console.log(data.emotional_response);
          // Navigate to response page with the data
          const searchParams = new URLSearchParams({
            transcription: data.transcription,
            vad_scores: JSON.stringify(data.vad_scores),
            emotional_response: JSON.stringify(data.emotional_response),
          });

          router.push(`/response?${searchParams.toString()}`);

          // Update state with transcription
          setTranscription(data.transcription);

          // Create a new audio element for streaming
          const audio = new Audio(`http://localhost:8000/${data.tts_file}`);
          audioRef.current = audio;

          // Set up event listeners
          audio.onended = () => {
            setIsPlaying(false);
            // Clean up the audio URL after playback
            if (audioUrl) {
              URL.revokeObjectURL(audioUrl);
              setAudioUrl(null);
            }
          };

          // Start playing automatically
          audio
            .play()
            .then(() => {
              setIsPlaying(true);
            })
            .catch((error) => {
              console.error("Error playing audio:", error);
            });
        } catch (error) {
          console.error("Error processing audio:", error);
        } finally {
          setIsUploading(false);
        }
      };

      mediaRecorder.start(1000);
      setIsRecording(true);
    } catch (error) {
      console.error("Error starting recording:", error);
      setIsRecording(false);
    }
  }, [router]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream
        .getTracks()
        .forEach((track) => track.stop());
      setIsRecording(false);
    }
  }, [isRecording]);

  const togglePlayback = useCallback(() => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  }, [isPlaying]);

  const handleButtonClick = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  return (
    <div className="flex flex-col items-center z-10 bg-white/20 backdrop-blur-sm p-10 rounded-2xl shadow-lg mt-8">
      <h1 className="text-4xl font-bold text-gray-800 mb-2">
        Human-like Voice Assistant
      </h1>
      <p className="text-lg text-gray-600 max-w-md text-center">
        Experience conversations with AI that feel remarkably human
      </p>

      {transcription && (
        <div className="mt-4 p-4 bg-white/30 rounded-lg max-w-md">
          <p className="text-gray-700">{transcription}</p>
        </div>
      )}

      <div className="flex items-center gap-4 mt-4">
        <motion.button
          className={`px-6 py-3 rounded-full text-white font-medium shadow-lg flex items-center gap-2 ${
            isRecording
              ? "bg-red-500 hover:bg-red-600"
              : "bg-gradient-to-r from-blue-400 via-purple-400 to-green-400"
          }`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleButtonClick}
          disabled={isUploading || isPlaying}
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

        {audioRef.current && (
          <motion.button
            className="px-6 py-3 rounded-full text-white font-medium shadow-lg flex items-center gap-2 bg-purple-500 hover:bg-purple-600"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={togglePlayback}
          >
            {isPlaying ? (
              <>
                <Pause className="w-5 h-5" />
                Pause
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Play Response
              </>
            )}
          </motion.button>
        )}
      </div>
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

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}
