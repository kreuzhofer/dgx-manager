export interface NetInterfaceSample {
  name: string;
  rxBytesPerSec: number;
  txBytesPerSec: number;
}

export interface DiskDeviceSample {
  name: string;
  readBytesPerSec: number;
  writeBytesPerSec: number;
}

export interface MemorySample {
  memTotalMb: number;
  memAvailableMb: number;
  memCachedMb: number;
  swapTotalMb: number;
  swapUsedMb: number;
}

export interface PressureSample {
  memorySomeAvg10: number | null;
  ioSomeAvg10: number | null;
  cpuSomeAvg10: number | null;
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
  diskDevices?: DiskDeviceSample[];
  memory?: MemorySample;
  pressure?: PressureSample;
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
