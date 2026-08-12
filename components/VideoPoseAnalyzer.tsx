"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils,
  type PoseLandmarkerResult
} from "@mediapipe/tasks-vision";
import {
  calculatePhaseMetrics,
  detectShotPhases,
  metricDeviationSummary,
  type Handedness,
  type PhaseMetrics,
  type PoseFrame
} from "@/lib/poseMath";

type CoachResponse = {
  shotFormSummary: string;
  primaryMechanicalError: string;
  correctiveDrillPlan: string[];
};

const CONNECTOR_LINE_WIDTH = 4;
const MAX_FRAME_BUFFER = 600;

const SHOULDER_CONNECTIONS: [number, number][] = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28]
];

export function VideoPoseAnalyzer() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameBufferRef = useRef<PoseFrame[]>([]);
  const animationFrameRef = useRef<number>();
  const poseRef = useRef<PoseLandmarker | null>(null);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [loadingModel, setLoadingModel] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [shootingSide, setShootingSide] = useState<Handedness>("right");
  const [metrics, setMetrics] = useState<PhaseMetrics | null>(null);
  const [coach, setCoach] = useState<CoachResponse | null>(null);

  const metricCards = useMemo(() => {
    if (!metrics) {
      return [];
    }

    return [
      {
        label: "Stance Width Ratio",
        value: metrics.stanceWidthRatio.toFixed(2),
        ok: metrics.inRange.stanceWidthRatio,
        target: "0.90 - 1.20"
      },
      {
        label: "Knee Dip Angle",
        value: `${metrics.kneeDipAngle.toFixed(1)}°`,
        ok: metrics.inRange.kneeDipAngle,
        target: "110° - 125°"
      },
      {
        label: "Set Angle",
        value: `${metrics.setAngle.toFixed(1)}°`,
        ok: metrics.inRange.setAngle,
        target: "85° - 95°"
      },
      {
        label: "Elbow Flare Offset",
        value: `${metrics.elbowFlareOffsetPercent.toFixed(1)}%`,
        ok: metrics.inRange.elbowFlareOffsetPercent,
        target: "< 5%"
      },
      {
        label: "Release Above Nose",
        value: metrics.releaseAboveNose ? "Yes" : "No",
        ok: metrics.inRange.releaseAboveNose,
        target: "Yes"
      }
    ];
  }, [metrics]);

  useEffect(() => {
    let mounted = true;

    const setupPoseLandmarker = async () => {
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        const poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numPoses: 1
        });

        if (!mounted) {
          poseLandmarker.close();
          return;
        }

        poseRef.current = poseLandmarker;
      } catch {
        setErrorMessage("Unable to load pose detector. Please refresh and try again.");
      } finally {
        if (mounted) {
          setLoadingModel(false);
        }
      }
    };

    void setupPoseLandmarker();

    return () => {
      mounted = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      poseRef.current?.close();
    };
  }, []);

  const resizeCanvasToVideo = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) {
      return;
    }

    const { videoWidth, videoHeight } = video;
    if (!videoWidth || !videoHeight) {
      return;
    }

    canvas.width = videoWidth;
    canvas.height = videoHeight;
  }, []);

  const drawPoseResult = useCallback(
    (result: PoseLandmarkerResult) => {
      const canvas = canvasRef.current;
      if (!canvas) {
        return;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        return;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const landmarks = result.landmarks[0];
      if (!landmarks) {
        return;
      }

      const drawingUtils = new DrawingUtils(ctx);

      for (const [start, end] of SHOULDER_CONNECTIONS) {
        const isArmConnection = [13, 14, 15, 16].includes(start) || [13, 14, 15, 16].includes(end);
        const isLegConnection = [23, 24, 25, 26, 27, 28].includes(start) && [23, 24, 25, 26, 27, 28].includes(end);

        const shouldFlagArm = metrics ? !metrics.inRange.elbowFlareOffsetPercent || !metrics.inRange.setAngle : false;
        const shouldFlagLeg = metrics ? !metrics.inRange.kneeDipAngle : false;

        const color =
          (isArmConnection && shouldFlagArm) || (isLegConnection && shouldFlagLeg) ? "#ef4444" : "#22c55e";

        drawingUtils.drawConnectors(landmarks, [{ start, end }], { color, lineWidth: CONNECTOR_LINE_WIDTH });
      }

      drawingUtils.drawLandmarks(landmarks, {
        color: "#e2e8f0",
        lineWidth: 1,
        radius: 2
      });
    },
    [metrics]
  );

  const sendToCoach = useCallback(async (calculatedMetrics: PhaseMetrics) => {
    try {
      const response = await fetch("/api/coach", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          metrics: calculatedMetrics,
          deviations: metricDeviationSummary(calculatedMetrics)
        })
      });

      if (!response.ok) {
        throw new Error("Failed to generate drills");
      }

      const payload = (await response.json()) as CoachResponse;
      setCoach(payload);
    } catch {
      setCoach({
        shotFormSummary: "Coach service is unavailable right now.",
        primaryMechanicalError: "Unable to evaluate via AI",
        correctiveDrillPlan: [
          "Re-upload the clip and verify a clear side profile view.",
          "Focus on one correction at a time: dip depth, set angle, then release.",
          "Retry AI coach generation once network access is stable."
        ]
      });
    }
  }, []);

  const processBufferedFrames = useCallback(() => {
    const phases = detectShotPhases(frameBufferRef.current, shootingSide);
    if (!phases) {
      return;
    }

    const calculated = calculatePhaseMetrics(phases, shootingSide);
    setMetrics(calculated);
    void sendToCoach(calculated);
  }, [sendToCoach, shootingSide]);

  const runFrameLoop = useCallback(() => {
    const video = videoRef.current;
    const pose = poseRef.current;

    if (!video || !pose) {
      return;
    }

    if (video.paused || video.ended) {
      setProcessing(false);
      processBufferedFrames();
      return;
    }

    const result = pose.detectForVideo(video, performance.now());
    if (result.landmarks.length > 0) {
      frameBufferRef.current.push({
        timestampMs: performance.now(),
        landmarks: result.landmarks[0]
      });

      if (frameBufferRef.current.length > MAX_FRAME_BUFFER) {
        frameBufferRef.current.shift();
      }
    }

    drawPoseResult(result);
    animationFrameRef.current = requestAnimationFrame(runFrameLoop);
  }, [drawPoseResult, processBufferedFrames]);

  const onVideoPlay = useCallback(() => {
    if (!poseRef.current) {
      return;
    }

    setProcessing(true);
    frameBufferRef.current = [];
    setCoach(null);
    setMetrics(null);
    resizeCanvasToVideo();
    animationFrameRef.current = requestAnimationFrame(runFrameLoop);
  }, [resizeCanvasToVideo, runFrameLoop]);

  const onUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (!file.type.includes("mp4") && !file.type.includes("quicktime")) {
      setErrorMessage("Please upload a valid .mp4 or .mov file.");
      return;
    }

    setErrorMessage(null);
    setMetrics(null);
    setCoach(null);

    const nextUrl = URL.createObjectURL(file);
    setVideoUrl(nextUrl);
  }, []);

  return (
    <section className="space-y-5 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <label className="flex items-center gap-3 text-sm font-medium text-slate-700">
          <span>Shooting video</span>
          <input
            type="file"
            accept="video/mp4,video/quicktime"
            className="rounded border border-slate-300 px-2 py-1 text-sm"
            onChange={onUpload}
          />
        </label>

        <label className="flex items-center gap-2 text-sm font-medium text-slate-700">
          Shooting Hand
          <select
            className="rounded border border-slate-300 px-2 py-1"
            value={shootingSide}
            onChange={(event) => setShootingSide(event.target.value as Handedness)}
          >
            <option value="right">Right</option>
            <option value="left">Left</option>
          </select>
        </label>
      </div>

      {loadingModel && <p className="rounded bg-sky-50 p-3 text-sm text-sky-700">Loading MediaPipe model...</p>}
      {errorMessage && <p className="rounded bg-red-50 p-3 text-sm text-red-700">{errorMessage}</p>}

      <div className="relative overflow-hidden rounded-lg border border-slate-300 bg-black">
        <video
          ref={videoRef}
          src={videoUrl ?? undefined}
          controls
          playsInline
          className="w-full"
          onLoadedMetadata={resizeCanvasToVideo}
          onPlay={onVideoPlay}
        />
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
      </div>

      {processing && <p className="text-sm text-slate-600">Processing frames and tracking biomechanics...</p>}
      {!videoUrl && !loadingModel && (
        <p className="rounded bg-slate-50 p-3 text-sm text-slate-600">
          Upload a side-profile catch-and-shoot video to begin analysis.
        </p>
      )}

      {metricCards.length > 0 && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {metricCards.map((card) => (
            <article
              key={card.label}
              className={`rounded-lg border p-3 ${
                card.ok ? "border-emerald-200 bg-emerald-50" : "border-red-200 bg-red-50"
              }`}
            >
              <p className="text-xs uppercase tracking-wide text-slate-500">{card.label}</p>
              <p className="mt-1 text-lg font-semibold">{card.value}</p>
              <p className="text-xs text-slate-600">Target: {card.target}</p>
            </article>
          ))}
        </div>
      )}

      {coach && (
        <article className="space-y-3 rounded-lg border border-indigo-200 bg-indigo-50 p-4">
          <h2 className="text-lg font-semibold text-indigo-900">AI Coach</h2>
          <p>
            <span className="font-semibold">Summary:</span> {coach.shotFormSummary}
          </p>
          <p>
            <span className="font-semibold">Primary Error:</span> {coach.primaryMechanicalError}
          </p>
          <div>
            <p className="font-semibold">3-Step Corrective Drill Plan:</p>
            <ol className="list-decimal space-y-1 pl-5">
              {coach.correctiveDrillPlan.map((drillStep, index) => (
                <li key={`${index}-${drillStep}`}>{drillStep}</li>
              ))}
            </ol>
          </div>
        </article>
      )}
    </section>
  );
}
