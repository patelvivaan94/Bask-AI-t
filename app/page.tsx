import { VideoPoseAnalyzer } from "@/components/VideoPoseAnalyzer";

export default function HomePage() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-6">
      <h1 className="text-3xl font-bold">Basketball Shot Biomechanics Analysis</h1>
      <VideoPoseAnalyzer />
    </main>
  );
}
