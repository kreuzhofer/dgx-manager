import type { Metadata } from "next";
import "./globals.css";
import { TopNav } from "@/components/top-nav";
import { Toaster } from "sonner";
import { DeploymentPullToast } from "@/components/deployment-pull-toast";

export const metadata: Metadata = {
  title: "DGX Manager",
  description: "Manage and monitor DGX Spark cluster",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 min-h-screen">
        <TopNav />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
        <Toaster theme="dark" position="bottom-right" richColors closeButton />
        <DeploymentPullToast />
      </body>
    </html>
  );
}
