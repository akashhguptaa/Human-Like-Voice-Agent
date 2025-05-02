import React, { useRef, useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import { Mic, Square } from "lucide-react";
import { AudioRecorder } from "@/utils/audioRecorder";

export const MainContent = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const audioRecorderRef = useRef<AudioRecorder | null>(null);

  useEffect(() => {
    audioRecorderRef.current = new AudioRecorder({
      onProcessingStart: () => setIsUploading(true),
      onProcessingEnd: () => setIsUploading(false),
      onError: (error) => {
        console.error("Recording error:", error);
        setIsUploading(false);
        setIsRecording(false);
      },
      onMessage: (data) => {
        console.log("Received message:", data);
      },
    });

    return () => {
      audioRecorderRef.current?.cleanup();
    };
  }, []);

  const handleButtonClick = useCallback(async () => {
    if (!audioRecorderRef.current) return;

    if (isRecording) {
      audioRecorderRef.current.stopRecording();
      setIsRecording(false);
    } else {
      await audioRecorderRef.current.startRecording();
      setIsRecording(true);
    }
  }, [isRecording]);

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
