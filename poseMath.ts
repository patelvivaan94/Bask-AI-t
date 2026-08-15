export type Handedness = "right" | "left";

export type PoseLandmark = {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
};

export type PoseFrame = {
  timestampMs: number;
  landmarks: PoseLandmark[];
};

export type ShotPhases = {
  dipFrame?: PoseFrame;
  setFrame?: PoseFrame;
  releaseFrame?: PoseFrame;
  dipFrameIndex: number | null;
  setPointFrameIndex: number | null;
  releaseFrameIndex: number | null;
  dipTimestamp: number | null;
  setTimestamp: number | null;
  releaseTimestamp: number | null;
};

export type PhaseMetrics = {
  stanceWidthRatio: number;
  kneeDipAngle: number;
  setAngle: number;
  elbowOffsetRatio: number; // e.g. 0.02 = 2%
  elbowFlareOffsetPercent: number; // e.g. 2.0 = 2%
  isHighRelease: boolean;
  releaseAboveNose: boolean;
  inRange: {
    stanceWidthRatio: boolean;
    kneeDipAngle: boolean;
    setAngle: boolean;
    elbowOffsetRatio: boolean;
    elbowFlareOffsetPercent: boolean;
    isHighRelease: boolean;
    releaseAboveNose: boolean;
  };
};

export type DeviationSummary = {
  hasErrors: boolean;
  issues: string[];
  metrics: PhaseMetrics;
};

function distance2D(p1: PoseLandmark, p2: PoseLandmark): number {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function angle3Point(a: PoseLandmark, b: PoseLandmark, c: PoseLandmark): number {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };

  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAB = Math.sqrt(ab.x * ab.x + ab.y * ab.y);
  const magCB = Math.sqrt(c.x * c.x + cb.y * cb.y);

  if (magAB === 0 || magCB === 0) return 0;

  const cosAngle = Math.max(-1, Math.min(1, dot / (magAB * magCB)));
  return (Math.acos(cosAngle) * 180) / Math.PI;
}

export function detectShotPhases(
  frames: PoseFrame[],
  handedness: Handedness
): ShotPhases | null {
  if (frames.length < 3) {
    return null;
  }

  const isRight = handedness === "right";
  const shoulderIdx = isRight ? 12 : 11;
  const wristIdx = isRight ? 16 : 15;
  const hipIdx = isRight ? 24 : 23;
  const kneeIdx = isRight ? 26 : 25;

  let dipFrame = frames[0];
  let dipIndex = 0;
  let maxY = -Infinity;

  frames.forEach((frame, idx) => {
    const hip = frame.landmarks[hipIdx];
    const knee = frame.landmarks[kneeIdx];
    if (hip && knee) {
      const currentY = (hip.y + knee.y) / 2;
      if (currentY > maxY) {
        maxY = currentY;
        dipFrame = frame;
        dipIndex = idx;
      }
    }
  });

  let releaseFrame = frames[frames.length - 1];
  let releaseIndex = frames.length - 1;
  let minY = Infinity;

  for (let i = dipIndex; i < frames.length; i++) {
    const wrist = frames[i].landmarks[wristIdx];
    if (wrist && wrist.y < minY) {
      minY = wrist.y;
      releaseFrame = frames[i];
      releaseIndex = i;
    }
  }

  let setIndex = Math.floor((dipIndex + releaseIndex) / 2);
  let setFrame = frames[setIndex] || frames[0];

  for (let i = dipIndex; i <= releaseIndex; i++) {
    const frame = frames[i];
    const wrist = frame.landmarks[wristIdx];
    const shoulder = frame.landmarks[shoulderIdx];

    if (wrist && shoulder) {
      if (Math.abs(wrist.y - shoulder.y) < 0.1) {
        setFrame = frame;
        setIndex = i;
        break;
      }
    }
  }

  return {
    dipFrame,
    setFrame,
    releaseFrame,
    dipFrameIndex: dipIndex,
    setPointFrameIndex: setIndex,
    releaseFrameIndex: releaseIndex,
    dipTimestamp: dipFrame ? dipFrame.timestampMs / 1000 : null,
    setTimestamp: setFrame ? setFrame.timestampMs / 1000 : null,
    releaseTimestamp: releaseFrame ? releaseFrame.timestampMs / 1000 : null,
  };
}

export function calculatePhaseMetrics(
  phases: ShotPhases,
  handedness: Handedness
): PhaseMetrics {
  const isRight = handedness === "right";
  const shoulderIdx = isRight ? 12 : 11;
  const elbowIdx = isRight ? 14 : 13;
  const wristIdx = isRight ? 16 : 15;
  const hipIdx = isRight ? 24 : 23;
  const kneeIdx = isRight ? 26 : 25;
  const ankleIdx = isRight ? 28 : 27;

  const leftAnkleIdx = 27;
  const rightAnkleIdx = 28;
  const leftShoulderIdx = 11;
  const rightShoulderIdx = 12;

  const stanceFrame = phases.dipFrame;
  const lAnkle = stanceFrame?.landmarks[leftAnkleIdx];
  const rAnkle = stanceFrame?.landmarks[rightAnkleIdx];
  const lShoulder = stanceFrame?.landmarks[leftShoulderIdx];
  const rShoulder = stanceFrame?.landmarks[rightShoulderIdx];

  let stanceWidthRatio = 1.05;
  if (lAnkle && rAnkle && lShoulder && rShoulder) {
    const ankleDist = distance2D(lAnkle, rAnkle);
    const shoulderDist = distance2D(lShoulder, rShoulder);
    if (shoulderDist > 0) {
      stanceWidthRatio = ankleDist / shoulderDist;
    }
  }

  const dipHip = phases.dipFrame?.landmarks[hipIdx];
  const dipKnee = phases.dipFrame?.landmarks[kneeIdx];
  const dipAnkle = phases.dipFrame?.landmarks[ankleIdx];

  let kneeDipAngle = 118;
  if (dipHip && dipKnee && dipAnkle) {
    kneeDipAngle = angle3Point(dipHip, dipKnee, dipAnkle);
  }

  const setShoulder = phases.setFrame?.landmarks[shoulderIdx];
  const setElbow = phases.setFrame?.landmarks[elbowIdx];
  const setWrist = phases.setFrame?.landmarks[wristIdx];

  let setAngle = 90;
  if (setShoulder && setElbow && setWrist) {
    setAngle = angle3Point(setShoulder, setElbow, setWrist);
  }

  let elbowFlareOffsetPercent = 2.0;
  if (setShoulder && setElbow && lShoulder && rShoulder) {
    const shoulderWidth = distance2D(lShoulder, rShoulder);
    if (shoulderWidth > 0) {
      const dx = Math.abs(setElbow.x - setShoulder.x);
      elbowFlareOffsetPercent = (dx / shoulderWidth) * 100;
    }
  }

  const noseIdx = 0;
  const relWrist = phases.releaseFrame?.landmarks[wristIdx];
  const relNose = phases.releaseFrame?.landmarks[noseIdx];

  let releaseAboveNose = true;
  if (relWrist && relNose) {
    releaseAboveNose = relWrist.y < relNose.y;
  }

  const elbowOffsetRatio = elbowFlareOffsetPercent / 100;
  const isHighRelease = releaseAboveNose;

  const inRange = {
    stanceWidthRatio: stanceWidthRatio >= 0.9 && stanceWidthRatio <= 1.2,
    kneeDipAngle: kneeDipAngle >= 110 && kneeDipAngle <= 125,
    setAngle: setAngle >= 85 && setAngle <= 95,
    elbowOffsetRatio: elbowFlareOffsetPercent < 5.0,
    elbowFlareOffsetPercent: elbowFlareOffsetPercent < 5.0,
    isHighRelease,
    releaseAboveNose
  };

  return {
    stanceWidthRatio,
    kneeDipAngle,
    setAngle,
    elbowOffsetRatio,
    elbowFlareOffsetPercent,
    isHighRelease,
    releaseAboveNose,
    inRange
  };
}

export function metricDeviationSummary(metrics: PhaseMetrics): string[] {
  const deviations: string[] = [];

  if (!metrics.inRange.stanceWidthRatio) {
    deviations.push(
      `Stance width ratio (${metrics.stanceWidthRatio.toFixed(2)}) is outside the recommended 0.90 - 1.20 range.`
    );
  }

  if (!metrics.inRange.kneeDipAngle) {
    deviations.push(
      `Knee dip angle (${metrics.kneeDipAngle.toFixed(1)}°) is outside optimal range (110° - 125°).`
    );
  }

  if (!metrics.inRange.setAngle) {
    deviations.push(
      `Set angle (${metrics.setAngle.toFixed(1)}°) deviates from target (85° - 95°).`
    );
  }

  if (!metrics.inRange.elbowFlareOffsetPercent) {
    deviations.push(
      `Elbow flare offset (${metrics.elbowFlareOffsetPercent.toFixed(1)}%) exceeds maximum 5% threshold.`
    );
  }

  if (!metrics.inRange.releaseAboveNose) {
    deviations.push(`Release point is below nose/eye line.`);
  }

  if (deviations.length === 0) {
    deviations.push("All biomechanical metrics are within ideal targets.");
  }

  return deviations;
}

export function getPresetMetrics(presetType: 'clean' | 'flared'): PhaseMetrics {
  if (presetType === 'clean') {
    return {
      stanceWidthRatio: 1.05,
      kneeDipAngle: 116.5,
      setAngle: 89.2,
      elbowOffsetRatio: 0.021,
      elbowFlareOffsetPercent: 2.1,
      isHighRelease: true,
      releaseAboveNose: true,
      inRange: {
        stanceWidthRatio: true,
        kneeDipAngle: true,
        setAngle: true,
        elbowOffsetRatio: true,
        elbowFlareOffsetPercent: true,
        isHighRelease: true,
        releaseAboveNose: true
      }
    };
  }

  // Flared
  return {
    stanceWidthRatio: 0.72,
    kneeDipAngle: 98.4,
    setAngle: 74.1,
    elbowOffsetRatio: 0.118,
    elbowFlareOffsetPercent: 11.8,
    isHighRelease: false,
    releaseAboveNose: false,
    inRange: {
      stanceWidthRatio: false,
      kneeDipAngle: false,
      setAngle: false,
      elbowOffsetRatio: false,
      elbowFlareOffsetPercent: false,
      isHighRelease: false,
      releaseAboveNose: false
    }
  };
}
