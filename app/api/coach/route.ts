import { NextResponse } from "next/server";
import { GoogleGenAI } from "@google/genai";
import type { PhaseMetrics } from "@/lib/poseMath";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type CoachRequestBody = {
  metrics: PhaseMetrics;
  deviations: string[];
};

type CoachResponseBody = {
  shotFormSummary: string;
  primaryMechanicalError: string;
  correctiveDrillPlan: string[];
};

const SYSTEM_PROMPT = `
You are an elite basketball biomechanics shooting coach.
Given pose-derived metrics and deviations, return strict JSON with keys:
shotFormSummary (string), primaryMechanicalError (string), correctiveDrillPlan (array of exactly 3 concise actionable drill steps).
Focus on practical corrections for a catch-and-shoot side-profile rep.
`.trim();

const parseCoachJson = (rawText: string): CoachResponseBody | null => {
  const normalized = rawText.trim();
  const jsonMatch = normalized.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    return null;
  }

  try {
    const parsed = JSON.parse(jsonMatch[0]) as Partial<CoachResponseBody>;
    if (
      typeof parsed.shotFormSummary === "string" &&
      typeof parsed.primaryMechanicalError === "string" &&
      Array.isArray(parsed.correctiveDrillPlan)
    ) {
      return {
        shotFormSummary: parsed.shotFormSummary,
        primaryMechanicalError: parsed.primaryMechanicalError,
        correctiveDrillPlan: parsed.correctiveDrillPlan.slice(0, 3).map((step) => String(step))
      };
    }
  } catch {
    return null;
  }

  return null;
};

export async function POST(request: Request) {
  let body: CoachRequestBody;

  try {
    body = (await request.json()) as CoachRequestBody;
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  if (!body.metrics || !Array.isArray(body.deviations)) {
    return NextResponse.json({ error: "Missing metrics or deviations." }, { status: 400 });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "Missing GEMINI_API_KEY." }, { status: 500 });
  }

  const prompt = `${SYSTEM_PROMPT}\n\nMetrics:\n${JSON.stringify(body.metrics, null, 2)}\n\nDeviations:\n${body.deviations.join("\n")}`;

  try {
    const ai = new GoogleGenAI({ apiKey });
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt
    });

    const rawText = response.text ?? "";
    const parsedCoach = parseCoachJson(rawText);
    if (parsedCoach) {
      return NextResponse.json(parsedCoach);
    }

    return NextResponse.json({
      shotFormSummary: rawText || "Shot analyzed with limited AI output.",
      primaryMechanicalError: "Could not parse a single primary error from model output.",
      correctiveDrillPlan: [
        "Use one-hand form shots to reinforce vertical forearm alignment.",
        "Practice dip-to-set rhythm with a 1-second pause at set point.",
        "Finish each rep with full follow-through and high release above eye line."
      ]
    });
  } catch {
    return NextResponse.json({ error: "Gemini request failed." }, { status: 502 });
  }
}
