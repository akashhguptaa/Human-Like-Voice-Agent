"use client";

import React, { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Mic } from "lucide-react"; // Import the mic icon from lucide-react

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.filter = "blur(5px)";

    const handleWindowResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      ctx.filter = "blur(3px)";
    };

    window.addEventListener("resize", handleWindowResize);

    const waves = [
      {
        color: "rgba(173, 216, 230, 0.5)",
        speed: 0.04,
        amplitude: 100,
        frequency: 0.005,
      },
      {
        color: "rgba(255, 182, 193, 0.4)",
        speed: 0.06,
        amplitude: 80,
        frequency: 0.007,
      },
      {
        color: "rgba(144, 238, 144, 0.4)",
        speed: 0.03,
        amplitude: 120,
        frequency: 0.004,
      },
    ];

    let time = 0;

    const drawWave = (
      ctx: CanvasRenderingContext2D,
      width: number,
      height: number,
      wave: {
        color: string;
        speed: number;
        amplitude: number;
        frequency: number;
      },
      time: number
    ) => {
      ctx.beginPath();
      ctx.moveTo(0, height);

      for (let x = 0; x < width; x += 5) {
        const y =
          Math.sin(x * wave.frequency + time * wave.speed) * wave.amplitude +
          height / 2;
        ctx.lineTo(x, y);
      }

      ctx.lineTo(width, height);
      ctx.lineTo(0, height);
      ctx.closePath();
      ctx.fillStyle = wave.color;
      ctx.fill();
    };

    const animate = () => {
      time += 0.08;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      waves.forEach((wave) => {
        drawWave(ctx, canvas.width, canvas.height, wave, time);
      });

      requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", handleWindowResize);
    };
  }, []);

  return (
    <main className="relative h-screen w-screen overflow-hidden flex flex-col items-center justify-center">
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full -z-10"
      />

      {/* Circular Glassmorphism Background */}
      <div className="relative z-10 flex items-center justify-center w-40 h-40 bg-white/20 backdrop-blur-md rounded-full shadow-lg">
        <Mic className="w-16 h-16 text-gray-800" />
      </div>

      <div className="flex flex-col items-center z-10 bg-white/20 backdrop-blur-sm p-10 rounded-2xl shadow-lg mt-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">MANAS-AI</h1>
        <p className="text-lg text-gray-600 max-w-md text-center">
          Experience conversations with AI that feel remarkably human
        </p>

        <motion.button
          className="mt-8 px-6 py-3 bg-gradient-to-r from-blue-400 via-purple-400 to-green-400 rounded-full text-white font-medium shadow-lg"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          Start Conversation
        </motion.button>
      </div>
    </main>
  );
}