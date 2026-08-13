# Bask-AI-t

Basketball shot biomechanics analyzer built with Next.js, MediaPipe Pose Landmarker, and Gemini.

## Setup

```bash
npm install
npm run dev
```

Create a `.env.local` file with:

```bash
GEMINI_API_KEY=your_api_key
```

## Implemented pipeline

- Video upload (`.mp4` / `.mov`) + side selection
- Real-time pose detection (`@mediapipe/tasks-vision`) with canvas skeleton overlay
- Auto-detected dip, set point, and release phases
- Biomechanics metric engine (`/lib/poseMath.ts`) for stance width, knee dip, set angle, elbow flare, and release height checks
- AI coaching route (`/app/api/coach/route.ts`) that sends metric deviations to Gemini (`gemini-2.5-flash`) and returns structured coaching output
