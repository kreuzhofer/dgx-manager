"use client";

import { useEffect, useRef, useState } from "react";

const LINKS: { href: string; label: string }[] = [
  { href: "/", label: "Overview" },
  { href: "/nodes", label: "Nodes" },
  { href: "/deployments", label: "Deployments" },
  { href: "/models", label: "Models" },
  { href: "/finetune", label: "Fine-tune" },
  { href: "/datasets", label: "Datasets" },
  { href: "/loadbalancer", label: "Load Balancer" },
  { href: "/benchmarks", label: "Benchmarks" },
  { href: "/settings", label: "Settings" },
];

function NavLink({
  href,
  children,
  onClick,
}: {
  href: string;
  children: React.ReactNode;
  onClick?: () => void;
}) {
  return (
    <a
      href={href}
      onClick={onClick}
      className="block px-3 py-2 rounded-md text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-700 transition-colors"
    >
      {children}
    </a>
  );
}

export function TopNav() {
  const [open, setOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  // Close on outside click + Escape, only while open. Listeners are attached
  // only when the menu is open to keep idle pages cheap.
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      const target = e.target as Node;
      if (panelRef.current?.contains(target) || buttonRef.current?.contains(target)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <nav className="bg-gray-900 border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <a href="/" className="flex items-center gap-2">
            <span className="text-xl font-bold text-green-400">DGX Manager</span>
          </a>

          {/* Desktop nav — visible >= md */}
          <div className="hidden md:flex items-center gap-1">
            {LINKS.map((l) => (
              <NavLink key={l.href} href={l.href}>
                {l.label}
              </NavLink>
            ))}
          </div>

          {/* Mobile hamburger — visible < md */}
          <button
            ref={buttonRef}
            type="button"
            aria-label={open ? "Close navigation menu" : "Open navigation menu"}
            aria-expanded={open}
            aria-controls="mobile-nav-panel"
            onClick={() => setOpen((v) => !v)}
            className="md:hidden inline-flex items-center justify-center p-2 rounded-md text-gray-300 hover:text-white hover:bg-gray-800 transition-colors"
          >
            {open ? (
              <svg className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            ) : (
              <svg className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="3" y1="6" x2="21" y2="6" />
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="18" x2="21" y2="18" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Mobile panel — only rendered when open */}
      {open && (
        <div
          id="mobile-nav-panel"
          ref={panelRef}
          className="md:hidden border-t border-gray-800 bg-gray-900"
        >
          <div className="max-w-7xl mx-auto px-2 sm:px-3 py-2 space-y-1">
            {LINKS.map((l) => (
              <NavLink key={l.href} href={l.href} onClick={() => setOpen(false)}>
                {l.label}
              </NavLink>
            ))}
          </div>
        </div>
      )}
    </nav>
  );
}
