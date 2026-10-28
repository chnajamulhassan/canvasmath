"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

interface DigitProbability {
  digit: number;
  probability: number;
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [prediction, setPrediction] = useState<string>("");
  const [confidence, setConfidence] = useState<string>("");
  const [probabilities, setProbabilities] = useState<DigitProbability[]>([]);
  const [loading, setLoading] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);

  const sessionRef = useRef<ort.InferenceSession | null>(null);

  // Load ONNX model
  useEffect(() => {
    const loadModel = async () => {
      try {
        const response = await fetch("/model/canvasmath.onnx");
        const modelArrayBuffer = await response.arrayBuffer();
        sessionRef.current =
          await ort.InferenceSession.create(modelArrayBuffer);
        console.log("ONNX model loaded successfully");
      } catch (error) {
        console.error("Failed to load ONNX model:", error);
        setPrediction("Model loading failed");
      }
    };
    loadModel();
  }, []);

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 22;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }, []);

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    setIsDrawing(true);
    const rect = canvas.getBoundingClientRect();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction("");
    setConfidence("");
    setProbabilities([]);
  };

  const predictDigit = async () => {
    const canvas = canvasRef.current;
    const session = sessionRef.current;

    if (!canvas || !session) {
      setPrediction("Model loading...");
      return;
    }

    try {
      setLoading(true);
      setPrediction("");
      setConfidence("");
      setProbabilities([]);

      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = 28;
      tempCanvas.height = 28;
      const tempCtx = tempCanvas.getContext("2d");
      if (!tempCtx) return;

      tempCtx.fillStyle = "black";
      tempCtx.fillRect(0, 0, 28, 28);
      tempCtx.drawImage(canvas, 0, 0, 28, 28);

      const imageData = tempCtx.getImageData(0, 0, 28, 28);
      const pixels = imageData.data;
      const inputData = new Float32Array(28 * 28);

      for (let i = 0; i < 28 * 28; i++) {
        const r = pixels[i * 4];
        const normalizedPixel = r / 255.0;
        inputData[i] = (normalizedPixel - 0.1307) / 0.3081;
      }

      const inputTensor = new ort.Tensor("float32", inputData, [1, 1, 28, 28]);
      const outputs = await session.run({
        [session.inputNames[0]]: inputTensor,
      });
      const scores = outputs[session.outputNames[0]].data as Float32Array;

      let maxScore = scores[0];
      for (let i = 1; i < scores.length; i++) {
        if (scores[i] > maxScore) maxScore = scores[i];
      }

      const expScores = Array.from(scores).map((s) => Math.exp(s - maxScore));
      const sumExp = expScores.reduce((sum, val) => sum + val, 0);
      const probs = expScores.map((val) => val / sumExp);

      let predictedClass = 0;
      let maxProb = probs[0];
      const probList: DigitProbability[] = [];

      for (let i = 0; i < probs.length; i++) {
        probList.push({ digit: i, probability: probs[i] * 100 });
        if (probs[i] > maxProb) {
          maxProb = probs[i];
          predictedClass = i;
        }
      }

      setPrediction(String(predictedClass));
      setConfidence(`${(maxProb * 100).toFixed(2)}%`);
      setProbabilities(probList);
    } catch (error) {
      console.error("Prediction error:", error);
      setPrediction("Failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 selection:bg-indigo-500 selection:text-white relative overflow-hidden">
      {/* Background Ambient Glow Effects */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="mx-auto max-w-5xl px-6 py-12 space-y-12 relative z-10">
        {/* Header */}
        <section className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold tracking-wide uppercase shadow-inner">
            <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
            Deep Learning Semester Project
          </div>
          <h1 className="text-5xl sm:text-6xl font-black tracking-tight bg-gradient-to-r from-white via-slate-200 to-indigo-400 bg-clip-text text-transparent">
            CanvasMath
          </h1>
          <p className="text-lg font-medium text-indigo-300">
            Handwritten Digit Recognition Web Application
          </p>
          <p className="max-w-2xl mx-auto text-slate-400 text-sm sm:text-base leading-relaxed">
            Draw any digit from 0–9 on the canvas below and experience real-time
            browser-based deep learning inference powered by ONNX Runtime Web.
          </p>
        </section>

        {/* Live Prediction Studio Card */}
        <section className="mx-auto max-w-3xl rounded-3xl bg-slate-900/70 backdrop-blur-xl p-6 sm:p-8 shadow-2xl border border-slate-800/80">
          <div className="flex items-center justify-between mb-6 pb-4 border-b border-slate-800/80">
            <h2 className="text-xl font-extrabold text-white flex items-center gap-2">
              <span className="text-indigo-400">⚡</span> Live Prediction Studio
            </h2>
            <span className="text-xs font-mono text-slate-400 bg-slate-950 px-3 py-1 rounded-full border border-slate-800">
              28x28 Tensor Input
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
            {/* Left Column: Canvas & Controls */}
            <div className="flex flex-col items-center space-y-4">
              <div className="relative group">
                <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-blue-600 rounded-2xl blur opacity-30 group-hover:opacity-60 transition duration-500 pointer-events-none" />
                <canvas
                  ref={canvasRef}
                  width={220}
                  height={220}
                  className="relative rounded-2xl border-2 border-slate-700/70 bg-slate-950 cursor-crosshair shadow-inner block"
                  onMouseDown={startDrawing}
                  onMouseMove={draw}
                  onMouseUp={stopDrawing}
                  onMouseLeave={stopDrawing}
                />
              </div>

              <div className="flex gap-3 w-full max-w-[220px]">
                <button
                  onClick={clearCanvas}
                  className="flex-1 rounded-xl bg-slate-800/80 hover:bg-slate-700 px-4 py-2.5 font-semibold text-slate-200 transition text-xs border border-slate-700/50 cursor-pointer active:scale-95"
                >
                  Clear
                </button>
                <button
                  onClick={predictDigit}
                  disabled={loading}
                  className="flex-1 rounded-xl bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 px-4 py-2.5 font-semibold text-white disabled:opacity-50 transition shadow-lg shadow-indigo-600/30 text-xs border border-indigo-400/20 cursor-pointer active:scale-95"
                >
                  {loading ? "Analyzing..." : "Predict"}
                </button>
              </div>
            </div>

            {/* Right Column: Results & Probabilities */}
            <div className="space-y-4">
              <div className="rounded-2xl bg-slate-950/80 p-5 text-center border border-slate-800/80 relative overflow-hidden">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Predicted Digit
                </p>
                <p className="mt-2 text-6xl font-black bg-gradient-to-b from-indigo-400 to-blue-500 bg-clip-text text-transparent">
                  {prediction || "—"}
                </p>
                {confidence && (
                  <div className="mt-2 inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-bold">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                    {confidence} Confidence
                  </div>
                )}
              </div>

              {probabilities.length > 0 ? (
                <div>
                  <h3 className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider text-center">
                    Class Probabilities (0–9)
                  </h3>
                  <div className="grid grid-cols-5 gap-2">
                    {probabilities.map((item) => (
                      <div
                        key={item.digit}
                        className={`rounded-xl p-2 text-center transition border text-xs ${
                          Number(prediction) === item.digit
                            ? "bg-indigo-950/60 border-indigo-500 text-indigo-300 font-bold shadow-md shadow-indigo-500/20"
                            : "bg-slate-950/40 border-slate-800/80 text-slate-400"
                        }`}
                      >
                        <div>{item.digit}</div>
                        <div className="font-semibold font-mono">
                          {item.probability.toFixed(0)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center text-xs text-slate-500 py-8 bg-slate-950/40 rounded-2xl border border-slate-800/60">
                  Draw any digit & click predict to view confidence
                  distribution.
                </div>
              )}
            </div>
          </div>
        </section>

        {/* About Dataset */}
        <section className="rounded-3xl bg-slate-900/70 backdrop-blur-xl p-8 border border-slate-800/80 shadow-xl">
          <h2 className="text-2xl font-bold mb-4 text-indigo-400 flex items-center gap-2">
            <span>📊</span> About the Dataset
          </h2>
          <p className="text-slate-300 leading-relaxed mb-4 text-sm sm:text-base">
            This project utilizes the benchmark{" "}
            <strong className="text-white">EMNIST Digits</strong> dataset,
            specifically formatted for robust handwritten character and digit
            recognition.
          </p>
          <ul className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm text-slate-300 font-medium">
            <li className="bg-slate-950/60 p-3.5 rounded-2xl border border-slate-800/80">
              <strong className="text-white">Classes:</strong> 10 distinct
              classes (Digits 0–9)
            </li>
            <li className="bg-slate-950/60 p-3.5 rounded-2xl border border-slate-800/80">
              <strong className="text-white">Training Samples:</strong> 240,000
              images
            </li>
            <li className="bg-slate-950/60 p-3.5 rounded-2xl border border-slate-800/80">
              <strong className="text-white">Test Samples:</strong> 40,000
              images
            </li>
            <li className="bg-slate-950/60 p-3.5 rounded-2xl border border-slate-800/80">
              <strong className="text-white">Image Format:</strong> 28 × 28
              grayscale pixels
            </li>
          </ul>
        </section>

        {/* Model Architecture */}
        <section className="rounded-3xl bg-slate-900/70 backdrop-blur-xl p-8 border border-slate-800/80 shadow-xl">
          <h2 className="text-2xl font-bold mb-4 text-indigo-400 flex items-center gap-2">
            <span>🧠</span> Model Architecture
          </h2>
          <p className="text-slate-300 mb-4 text-sm sm:text-base">
            A custom Convolutional Neural Network (CNN) built with PyTorch and
            optimized for high-accuracy digit recognition:
          </p>
          <div className="bg-slate-950 p-5 rounded-2xl font-mono text-xs sm:text-sm text-slate-300 space-y-2 border border-slate-800/80 shadow-inner leading-relaxed">
            <p className="text-slate-400">Input (28 × 28 × 1)</p>
            <p className="text-indigo-400">
              ↓ Conv2D (1 → 32, 3×3, padding=1) + ReLU + MaxPool (2×2)
            </p>
            <p className="text-indigo-400">
              ↓ Conv2D (32 → 64, 3×3, padding=1) + ReLU + MaxPool (2×2)
            </p>
            <p className="text-blue-400">
              ↓ Fully Connected (3136 → 128) + ReLU + Dropout (0.5)
            </p>
            <p className="text-emerald-400">
              ↓ Fully Connected (128 → 10 Output Classes)
            </p>
          </div>
        </section>

        {/* Results */}
        <section className="rounded-3xl bg-slate-900/70 backdrop-blur-xl p-8 border border-slate-800/80 shadow-xl">
          <h2 className="text-2xl font-bold mb-4 text-indigo-400 flex items-center gap-2">
            <span>📈</span> Evaluation Results
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
            <div className="bg-slate-950/80 p-5 rounded-2xl border border-slate-800/80">
              <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
                Test Accuracy
              </p>
              <p className="text-2xl font-black text-emerald-400 mt-1">
                99.51%
              </p>
            </div>
            <div className="bg-slate-950/80 p-5 rounded-2xl border border-slate-800/80">
              <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
                Test Loss
              </p>
              <p className="text-2xl font-black text-blue-400 mt-1">0.0173</p>
            </div>
            <div className="bg-slate-950/80 p-5 rounded-2xl border border-slate-800/80">
              <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
                Precision
              </p>
              <p className="text-2xl font-black text-purple-400 mt-1">0.9951</p>
            </div>
            <div className="bg-slate-950/80 p-5 rounded-2xl border border-slate-800/80">
              <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
                F1-Score
              </p>
              <p className="text-2xl font-black text-indigo-400 mt-1">0.9951</p>
            </div>
          </div>
        </section>

        {/* Technical Details */}
        <section className="rounded-3xl bg-slate-900/70 backdrop-blur-xl p-8 border border-slate-800/80 shadow-xl">
          <h2 className="text-2xl font-bold mb-4 text-indigo-400 flex items-center gap-2">
            <span>🛠️</span> Technical Details
          </h2>
          <div className="flex flex-wrap gap-2.5">
            {[
              "Python",
              "PyTorch",
              "EMNIST",
              "ONNX",
              "ONNX Runtime Web",
              "Next.js",
              "React",
              "Tailwind CSS",
              "Vercel",
            ].map((tech) => (
              <span
                key={tech}
                className="bg-slate-950 text-slate-300 px-4 py-2 rounded-xl text-xs font-semibold border border-slate-800/80 shadow-sm hover:border-indigo-500/50 transition cursor-default"
              >
                {tech}
              </span>
            ))}
          </div>
        </section>

        {/* Project Resources (Ab Clickable Links ke sath!) */}
        <section className="rounded-3xl bg-slate-900/70 backdrop-blur-xl p-8 border border-slate-800/80 shadow-xl">
          <h2 className="text-2xl font-bold mb-4 text-indigo-400 flex items-center gap-2">
            <span>📁</span> Project Resources
          </h2>
          <p className="text-slate-400 text-xs mb-6">
            Apne project ke links niche wale cards mein add karein (jaise GitHub
            repo, PDF report, presentation, YouTube demo).
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* GitHub Link */}
            <a
              href="https://github.com/your-username/canvasmath"
              target="_blank"
              rel="noopener noreferrer"
              className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800/80 hover:border-indigo-500/50 hover:bg-slate-950 transition group block"
            >
              <div className="flex items-center justify-between mb-1">
                <h3 className="font-bold text-white text-sm group-hover:text-indigo-400 transition">
                  GitHub Repository
                </h3>
                <span className="text-indigo-400 text-xs">↗</span>
              </div>
              <p className="text-xs text-slate-400">Source code & weights</p>
            </a>

            {/* Project Report Link */}
            <a
              href="https://drive.google.com/your-report-link"
              target="_blank"
              rel="noopener noreferrer"
              className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800/80 hover:border-indigo-500/50 hover:bg-slate-950 transition group block"
            >
              <div className="flex items-center justify-between mb-1">
                <h3 className="font-bold text-white text-sm group-hover:text-indigo-400 transition">
                  Project Report
                </h3>
                <span className="text-indigo-400 text-xs">↗</span>
              </div>
              <p className="text-xs text-slate-400">
                Comprehensive PDF document
              </p>
            </a>

            {/* PPT Presentation Link */}
            <a
              href="https://docs.google.com/your-ppt-link"
              target="_blank"
              rel="noopener noreferrer"
              className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800/80 hover:border-indigo-500/50 hover:bg-slate-950 transition group block"
            >
              <div className="flex items-center justify-between mb-1">
                <h3 className="font-bold text-white text-sm group-hover:text-indigo-400 transition">
                  PPT Presentation
                </h3>
                <span className="text-indigo-400 text-xs">↗</span>
              </div>
              <p className="text-xs text-slate-400">
                Slides with voiceover / notes
              </p>
            </a>

            {/* YouTube Demo Link */}
            <a
              href="https://youtube.com/your-demo-video"
              target="_blank"
              rel="noopener noreferrer"
              className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800/80 hover:border-indigo-500/50 hover:bg-slate-950 transition group block"
            >
              <div className="flex items-center justify-between mb-1">
                <h3 className="font-bold text-white text-sm group-hover:text-indigo-400 transition">
                  YouTube Video
                </h3>
                <span className="text-indigo-400 text-xs">↗</span>
              </div>
              <p className="text-xs text-slate-400">
                Demo walkthrough recording
              </p>
            </a>

            {/* Dataset Link */}
            <a
              href="https://www.nist.gov/itl/iad/image-group/emnist-dataset"
              target="_blank"
              rel="noopener noreferrer"
              className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800/80 hover:border-indigo-500/50 hover:bg-slate-950 transition group block"
            >
              <div className="flex items-center justify-between mb-1">
                <h3 className="font-bold text-white text-sm group-hover:text-indigo-400 transition">
                  Dataset Link
                </h3>
                <span className="text-indigo-400 text-xs">↗</span>
              </div>
              <p className="text-xs text-slate-400">Official EMNIST source</p>
            </a>
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center pb-8 border-t border-slate-900 pt-8 text-slate-500 text-xs space-y-1">
          <p className="font-medium text-slate-400">
            CanvasMath — Deep Learning Semester Project
          </p>
          <p>Built with PyTorch, ONNX Runtime Web, and Next.js</p>
        </footer>
      </div>
    </main>
  );
}
