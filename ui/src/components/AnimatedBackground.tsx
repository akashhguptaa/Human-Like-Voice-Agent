"use client";

import React, { useEffect, useRef } from "react";

interface Wave {
  color: string;
  speed: number;
  amplitude: number;
  frequency: number;
}

const waves: Wave[] = [
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

const drawWave = (
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  wave: Wave,
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

export const AnimatedBackground = () => {
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

    let time = 0;

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
    <canvas
      ref={canvasRef}
      className="absolute top-0 left-0 w-full h-full -z-10"
    />
  );
};
