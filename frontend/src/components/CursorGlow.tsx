import { useEffect, useRef } from "react";

export default function CursorGlow() {
  const dotRef   = useRef<HTMLDivElement>(null);
  const glowRef  = useRef<HTMLDivElement>(null);
  const trailRef = useRef<HTMLDivElement[]>([]);
  const pos      = useRef({ x: 0, y: 0 });
  const raf      = useRef<number>(0);

  useEffect(() => {
    const TRAIL = 8;
    const trail: { x: number; y: number }[] = Array(TRAIL).fill({ x: 0, y: 0 });

    // inject trail divs
    const container = document.body;
    const nodes: HTMLDivElement[] = [];
    for (let i = 0; i < TRAIL; i++) {
      const d = document.createElement("div");
      d.style.cssText = `
        position:fixed;pointer-events:none;z-index:9999;border-radius:50%;
        width:${6 - i * 0.5}px;height:${6 - i * 0.5}px;
        background:rgba(124,58,237,${0.6 - i * 0.07});
        box-shadow:0 0 ${8 - i}px rgba(124,58,237,${0.5 - i * 0.06});
        transform:translate(-50%,-50%);
        transition:opacity 0.1s;
      `;
      container.appendChild(d);
      nodes.push(d);
    }
    trailRef.current = nodes;

    function onMove(e: MouseEvent) {
      pos.current = { x: e.clientX, y: e.clientY };
      if (dotRef.current) {
        dotRef.current.style.left  = e.clientX + "px";
        dotRef.current.style.top   = e.clientY + "px";
      }
      if (glowRef.current) {
        glowRef.current.style.left = e.clientX + "px";
        glowRef.current.style.top  = e.clientY + "px";
      }
    }

    function tick() {
      trail.unshift({ ...pos.current });
      trail.length = TRAIL;
      nodes.forEach((n, i) => {
        const p = trail[i] ?? trail[trail.length - 1];
        n.style.left = p.x + "px";
        n.style.top  = p.y + "px";
      });
      raf.current = requestAnimationFrame(tick);
    }

    window.addEventListener("mousemove", onMove);
    raf.current = requestAnimationFrame(tick);

    return () => {
      window.removeEventListener("mousemove", onMove);
      cancelAnimationFrame(raf.current);
      nodes.forEach((n) => n.remove());
    };
  }, []);

  return (
    <>
      {/* outer glow ring */}
      <div
        ref={glowRef}
        className="fixed pointer-events-none z-[9998] rounded-full border border-accent/40"
        style={{
          width: 36, height: 36,
          transform: "translate(-50%,-50%)",
          transition: "left 0.12s ease, top 0.12s ease",
          boxShadow: "0 0 16px rgba(124,58,237,0.25)",
          background: "rgba(124,58,237,0.06)",
        }}
      />
      {/* center dot */}
      <div
        ref={dotRef}
        className="fixed pointer-events-none z-[9999] rounded-full bg-accent"
        style={{
          width: 5, height: 5,
          transform: "translate(-50%,-50%)",
          boxShadow: "0 0 8px rgba(124,58,237,0.9)",
        }}
      />
    </>
  );
}
