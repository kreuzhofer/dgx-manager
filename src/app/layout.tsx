import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "DGX Manager",
  description: "Manage and monitor DGX nodes",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
