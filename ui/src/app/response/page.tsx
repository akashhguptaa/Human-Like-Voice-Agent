"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useSearchParams } from "next/navigation";

// Sequential responses
const SEQUENTIAL_RESPONSES = [
  {
    response: {
      res_sentence_1: {
        text: "Good morning! I'm feeling great and ready to help with anything you need today.",
        vad: [0.7, 0.6, 0.8],
      },
    },
  },
  {
    response: {
      res_sentence_1: {
        text: "You sound really excited and happy! It's great to hear your enthusiasm—what can we explore next together?",
        vad: [0.9, 0.8, 0.7],
      },
      res_sentence_2: {
        text: "Feel free to ask me anything, and I'll respond with appropriate answers.",
        vad: [0.6, 0.75, 0.66],
      },
    },
  },
];

export default function ResponsePage() {
  const searchParams = useSearchParams();
  const transcription = searchParams.get("transcription") || "";
  const vadScores = JSON.parse(searchParams.get("vad_scores") || "{}");

  // Initialize state from localStorage or default to 0
  const [responseIndex, setResponseIndex] = useState(() => {
    if (typeof window !== "undefined") {
      const savedIndex = localStorage.getItem("responseIndex");
      return savedIndex ? parseInt(savedIndex) : 0;
    }
    return 0;
  });

  // Get the current response based on the index
  const emotionalResponse =
    SEQUENTIAL_RESPONSES[responseIndex] || SEQUENTIAL_RESPONSES[0];
  const responses = emotionalResponse?.response || {};

  // Update response index only when transcription changes
  useEffect(() => {
    if (transcription && transcription.trim() !== "") {
      const newIndex = (responseIndex + 1) % SEQUENTIAL_RESPONSES.length;
      setResponseIndex(newIndex);
      localStorage.setItem("responseIndex", newIndex.toString());
    }
  }, [transcription]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 p-8">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white/30 backdrop-blur-sm p-8 rounded-2xl shadow-lg"
        >
          <h1 className="text-3xl font-bold text-gray-800 mb-6">
            Conversation Analysis
          </h1>

          {/* User's Message Section */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-3">
              Your Message
            </h2>
            <div className="bg-white/40 p-4 rounded-lg">
              <p className="text-gray-700">{transcription}</p>
            </div>
          </div>

          {/* VAD Scores Section */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-3">
              VAD Analysis
            </h2>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-white/40 p-4 rounded-lg">
                <p className="text-sm text-gray-800">Arousal</p>
                <p className="text-xl font-medium text-gray-800">
                  {vadScores?.arousal?.toFixed(2) || "0.00"}
                </p>
              </div>
              <div className="bg-white/40 p-4 rounded-lg">
                <p className="text-sm text-gray-800">Dominance</p>
                <p className="text-xl font-medium text-gray-800">
                  {vadScores?.dominance?.toFixed(2) || "0.00"}
                </p>
              </div>
              <div className="bg-white/40 p-4 rounded-lg">
                <p className="text-sm text-gray-800">Valence</p>
                <p className="text-xl font-medium text-gray-800">
                  {vadScores?.valence?.toFixed(2) || "0.00"}
                </p>
              </div>
            </div>
          </div>

          {/* AI Response Section */}
          <div>
            <h2 className="text-xl font-semibold text-gray-800 mb-3">
              AI Response
            </h2>
            <div className="space-y-4">
              {Object.entries(responses).map(([key, value]: [string, any]) => (
                <motion.div
                  key={key}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="bg-white/40 p-6 rounded-lg"
                >
                  <p className="text-gray-700 mb-3">{value?.text || ""}</p>
                  <div className="grid grid-cols-3 gap-3">
                    <div className="bg-white/30 p-2 rounded">
                      <span className="text-gray-800">Arousal:</span>{" "}
                      <span className="text-gray-800">
                        {value?.vad?.[0]?.toFixed(2) || "0.00"}
                      </span>
                    </div>
                    <div className="bg-white/30 p-2 rounded">
                      <span className="text-gray-800">Dominance:</span>{" "}
                      <span className="text-gray-800">
                        {value?.vad?.[1]?.toFixed(2) || "0.00"}
                      </span>
                    </div>
                    <div className="bg-white/30 p-2 rounded">
                      <span className="text-gray-800">Valence:</span>{" "}
                      <span className="text-gray-800">
                        {value?.vad?.[2]?.toFixed(2) || "0.00"}
                      </span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
