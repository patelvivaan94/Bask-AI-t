import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Bask-AI | Shot Biomechanics & AI Coach",
  description: "Real-time basketball shot biomechanics analyzer and Gemini AI shooting coach"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased">{children}</body>
    </html>
  );
}
