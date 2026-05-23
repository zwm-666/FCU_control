import { useState, useEffect, useRef } from 'react';
import { MachineState, ChartDataPoint } from '../types';

const HISTORY_MAX_SECONDS = 600; // 10 minutes
const SAMPLE_INTERVAL_MS = 500;  // 2 Hz

interface UseChartHistoryReturn {
  history: ChartDataPoint[];
}

export function useChartHistory(
  isConnected: boolean,
  machine: MachineState,
): UseChartHistoryReturn {
  const [history, setHistory] = useState<ChartDataPoint[]>([]);

  // Keep a ref to the latest machine state to avoid stale closure inside setInterval
  const machineRef = useRef<MachineState>(machine);
  useEffect(() => {
    machineRef.current = machine;
  }, [machine]);

  // Stable ref for the chart's base timestamp so the x-axis never jumps back
  const startTimeRef = useRef<number | null>(null);

  useEffect(() => {
    if (!isConnected) {
      // Reset history and base time when disconnected
      setHistory([]);
      startTimeRef.current = null;
      return;
    }

    const interval = setInterval(() => {
      setHistory(prev => {
        const now = Date.now();

        // Drop points older than the rolling window
        const filtered = prev.filter(
          p => (now - p.timestamp) / 1000 <= HISTORY_MAX_SECONDS,
        );

        // Establish or recover the base time for the x-axis
        if (filtered.length === 0) {
          startTimeRef.current = now;
        } else if (startTimeRef.current === null) {
          startTimeRef.current = filtered[0].timestamp;
        }

        const baseTime = startTimeRef.current!;
        const m = machineRef.current;

        const newPoint: ChartDataPoint = {
          time: Number(((now - baseTime) / 1000).toFixed(1)),
          timestamp: now,
          voltage: Number(m.power.stackVoltage) || 0,
          current: Number(m.power.stackCurrent) || 0,
          temp: Number(m.sensors.stackTemp) || 0,
          h2Pressure: Number(m.sensors.h2CylinderPressure) || 0,
          stackPower: Number(m.power.stackPower) || 0,
        };

        return [...filtered, newPoint];
      });
    }, SAMPLE_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [isConnected]);

  return { history };
}
