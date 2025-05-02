import React from "react";
import { Mic } from "lucide-react";

export const MicrophoneIcon = () => {
  return (
    <div className="relative z-10 flex items-center justify-center w-40 h-40 bg-white/20 backdrop-blur-md rounded-full shadow-lg">
      <Mic className="w-16 h-16 text-gray-800" />
    </div>
  );
};
