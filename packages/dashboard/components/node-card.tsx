interface NodeMetric {
  gpuUtil: number;
  vramUsed: number;
  tps: number | null;
  temperature: number | null;
}

interface Node {
  id: string;
  name: string;
  ipAddress: string;
  status: string;
  gpuModel: string | null;
  vramTotal: number | null;
  metrics: NodeMetric[];
}

function MetricGauge({ label, value, max, unit }: { label: string; value: number; max: number; unit: string }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  const color = pct > 90 ? "bg-red-500" : pct > 70 ? "bg-yellow-500" : "bg-green-500";

  return (
    <div>
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{label}</span>
        <span>{value}{unit} / {max}{unit}</span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

const statusColors: Record<string, string> = {
  online: "bg-green-500",
  offline: "bg-red-500",
  new: "bg-gray-500",
};

export function NodeCard({ node }: { node: Node }) {
  const latest = node.metrics[0];
  const statusColor = statusColors[node.status] || "bg-gray-500";

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-semibold text-lg">{node.name}</h3>
          <p className="text-xs text-gray-500">{node.ipAddress}</p>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${statusColor}`} />
          <span className="text-xs text-gray-400 capitalize">{node.status}</span>
        </div>
      </div>

      {node.gpuModel && (
        <p className="text-xs text-gray-400 mb-3">{node.gpuModel}</p>
      )}

      {latest ? (
        <div className="space-y-2">
          <MetricGauge label="GPU" value={latest.gpuUtil} max={100} unit="%" />
          <MetricGauge label="VRAM" value={latest.vramUsed} max={node.vramTotal || 128000} unit=" MB" />
          {latest.temperature !== null && (
            <div className="text-xs text-gray-400 mt-2">
              Temp: {latest.temperature}°C
              {latest.tps !== null && <span className="ml-4">TPS: {latest.tps}</span>}
            </div>
          )}
        </div>
      ) : (
        <p className="text-xs text-gray-500 italic">No metrics yet</p>
      )}
    </div>
  );
}
