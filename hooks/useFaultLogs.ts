import { useState, useEffect } from 'react';
import { FaultLevel, FaultLog, FAULT_CODES } from '../types';

interface UseFaultLogsReturn {
  faultLogs: FaultLog[];
}

const MAX_FAULT_LOGS = 50;
// Minimum interval (ms) before the same fault code is logged again
const DEBOUNCE_MS = 2000;

export function useFaultLogs(
  faultCode: number,
  faultLevel: FaultLevel,
): UseFaultLogsReturn {
  const [faultLogs, setFaultLogs] = useState<FaultLog[]>([]);

  useEffect(() => {
    if (faultCode === 0) return;

    setFaultLogs(prev => {
      const lastLog = prev[0];
      const now = Date.now();

      // Deduplicate: skip if same code arrived within the debounce window
      if (lastLog && lastLog.code === faultCode && now - lastLog.id < DEBOUNCE_MS) {
        return prev;
      }

      const newLog: FaultLog = {
        id: now,
        time: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
        level: faultLevel,
        code: faultCode,
        description: FAULT_CODES[faultCode] ?? '未知故障',
      };

      return [newLog, ...prev].slice(0, MAX_FAULT_LOGS);
    });
  }, [faultCode, faultLevel]);

  return { faultLogs };
}
