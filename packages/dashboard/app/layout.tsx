import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DGX Manager",
  description: "Manage and monitor DGX Spark cluster",
};

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a href={href} className="px-3 py-2 rounded-md text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-700 transition-colors">
      {children}
    </a>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 min-h-screen">
        <nav className="bg-gray-900 border-b border-gray-800">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold text-green-400">DGX Manager</span>
              </div>
              <div className="flex items-center gap-1">
                <NavLink href="/">Overview</NavLink>
                <NavLink href="/nodes">Nodes</NavLink>
                <NavLink href="/deployments">Deployments</NavLink>
                <NavLink href="/finetune">Fine-tune</NavLink>
                <NavLink href="/datasets">Datasets</NavLink>
                <NavLink href="/loadbalancer">Load Balancer</NavLink>
                <NavLink href="/settings">Settings</NavLink>
              </div>
            </div>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
