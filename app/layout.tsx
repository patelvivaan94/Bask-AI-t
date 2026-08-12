import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Bask-AI-t",
  description: "Basketball shot biomechanics analyzer"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
