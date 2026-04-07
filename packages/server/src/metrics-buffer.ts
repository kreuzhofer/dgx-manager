export interface NetInterfaceSample {
  name: string;
  rxBytesPerSec: number;
  txBytesPerSec: number;
}

export interface MetricSample {
  timestamp: number;
  gpuUtil: number;
  vramUsed: number;
  temperature: number | null;
  tps: number | null;
  activeRequests: number | null;
  netInterfaces?: NetInterfaceSample[];
  rdmaInterfaces?: NetInterfaceSample[];
}

const MAX_SAMPLES = 720; // 1 hour at 5s intervals

class MetricsBuffer {
  private buffers = new Map<string, MetricSample[]>();

  push(nodeId: string, sample: MetricSample) {
    let buf = this.buffers.get(nodeId);
    if (!buf) {
      buf = [];
      this.buffers.set(nodeId, buf);
    }
    buf.push(sample);
    if (buf.length > MAX_SAMPLES) buf.shift();
  }

  getHistory(nodeId: string): MetricSample[] {
    return this.buffers.get(nodeId) || [];
  }

  remove(nodeId: string) {
    this.buffers.delete(nodeId);
  }
}

export const metricsBuffer = new MetricsBuffer();
