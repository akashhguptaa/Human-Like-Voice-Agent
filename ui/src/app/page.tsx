"use client";

import React from "react";
import { AnimatedBackground } from "@/components/AnimatedBackground";
import { MicrophoneIcon } from "@/components/MicrophoneIcon";
import { MainContent } from "@/components/MainContent";

export default function Home() {
  return (
    <main className="relative h-screen w-screen overflow-hidden flex flex-col items-center justify-center">
      <AnimatedBackground />
      <MicrophoneIcon />
      <MainContent />
    </main>
  );
}
