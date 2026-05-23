import { useRef, useState, useEffect } from 'react';
import { MachineState } from '../types';

interface UseDataLoggerReturn {
  isLogging: boolean;
  handleToggleLog: () => Promise<void>;
}

export function useDataLogger(machine: MachineState): UseDataLoggerReturn {
  const writableStreamRef = useRef<FileSystemWritableFileStream | null>(null);
  const bufferRef = useRef<string[]>([]);
  const [isLogging, setIsLogging] = useState(false);

  // Flush buffer to disk every 2 seconds
  useEffect(() => {
    const interval = setInterval(async () => {
      if (isLogging && writableStreamRef.current && bufferRef.current.length > 0) {
        try {
          const chunk = bufferRef.current.join('');
          bufferRef.current = [];
          await writableStreamRef.current.write(chunk);
        } catch (err) {
          console.error('Write error:', err);
        }
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [isLogging]);

  // Collect one CSV row on each machine state update
  useEffect(() => {
    if (!isLogging) return;

    const now = new Date().toLocaleString('zh-CN', { hour12: false });
    const line = [
      now,
      machine.power.stackVoltage.toFixed(1),
      machine.power.stackCurrent.toFixed(1),
      machine.sensors.stackTemp.toFixed(1),
      machine.sensors.h2InletPressure.toFixed(2),
      machine.power.dcfOutVoltage.toFixed(1),
      machine.power.dcfOutCurrent.toFixed(1),
      machine.io.fan1Duty,
      machine.io.faultCode,
    ].join(',') + '\n';

    bufferRef.current.push(line);
  }, [machine, isLogging]);

  const handleToggleLog = async () => {
    // --- Stop logging ---
    if (isLogging) {
      try {
        if (writableStreamRef.current) {
          if (bufferRef.current.length > 0) {
            await writableStreamRef.current.write(bufferRef.current.join(''));
            bufferRef.current = [];
          }
          await writableStreamRef.current.close();
          writableStreamRef.current = null;
        }
        setIsLogging(false);
      } catch (err) {
        console.error('Error stopping log:', err);
        alert('停止记录时发生错误，部分数据可能未保存。');
      }
      return;
    }

    // --- Start logging ---
    try {
      if (!('showSaveFilePicker' in window)) {
        alert(
          '当前浏览器不支持本地文件写入 API (File System Access API)。请使用 Chrome 或 Edge 桌面版。',
        );
        return;
      }

      const nowStr = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const handle = await window.showSaveFilePicker({
        suggestedName: `HMI_Log_${nowStr}.csv`,
        types: [
          {
            description: 'CSV Data Log',
            accept: { 'text/csv': ['.csv'] },
          },
        ],
      });

      const stream = await handle.createWritable();
      await stream.write(
        'Timestamp,Stack_Voltage(V),Stack_Current(A),Stack_Temp(C),' +
          'H2_Pressure(MPa),DCDC_Voltage(V),DCDC_Current(A),Fan1_Duty(%),Fault_Code\n',
      );

      writableStreamRef.current = stream;
      setIsLogging(true);
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      const message = err instanceof Error ? err.message : String(err);
      console.error('Failed to start logging:', err);
      alert('无法创建日志文件: ' + message);
    }
  };

  return { isLogging, handleToggleLog };
}
