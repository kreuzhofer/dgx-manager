"use client";

interface SparklineProps {
  data: number[];
  max: number;
  color: string;
  label: string;
  currentValue?: string;
  unit?: string;
  height?: number;
}

export function Sparkline({
  data,
  max,
  color,
  label,
  currentValue,
  unit = "",
  height = 48,
}: SparklineProps) {
  const width = 200;
  const padding = 2;
  const graphH = height - padding * 2;
  const graphW = width - padding * 2;

  const points = data.length > 1
    ? data.map((v, i) => {
        const x = padding + (i / (data.length - 1)) * graphW;
        const y = padding + graphH - (Math.min(v, max) / max) * graphH;
        return `${x},${y}`;
      })
    : [];

  const linePath = points.length > 0 ? `M${points.join("L")}` : "";
  const areaPath =
    points.length > 0
      ? `${linePath}L${padding + graphW},${padding + graphH}L${padding},${padding + graphH}Z`
      : "";

  return (
    <div className="min-w-0">
      <div className="flex justify-between items-baseline mb-0.5">
        <span className="text-[10px] text-gray-500 uppercase tracking-wide">{label}</span>
        <span className="text-xs font-mono text-gray-300">
          {currentValue ?? (data.length > 0 ? `${Math.round(data[data.length - 1])}` : "—")}
          {currentValue === undefined && data.length > 0 && (
            <span className="text-gray-500">{unit}</span>
          )}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="w-full rounded"
        style={{ height: `${height}px` }}
      >
        <rect
          x={0} y={0} width={width} height={height}
          rx={4} fill="rgba(0,0,0,0.2)"
        />
        {areaPath && (
          <path d={areaPath} fill={color} opacity={0.12} />
        )}
        {linePath && (
          <path
            d={linePath}
            fill="none"
            stroke={color}
            strokeWidth={1.5}
            vectorEffect="non-scaling-stroke"
          />
        )}
        {data.length === 0 && (
          <text
            x={width / 2} y={height / 2}
            textAnchor="middle" dominantBaseline="middle"
            fill="#6b7280" fontSize={10}
          >
            No data
          </text>
        )}
      </svg>
    </div>
  );
}
