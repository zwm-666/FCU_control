
import React, { useState, useEffect } from 'react';
import { INITIAL_MACHINE_STATE, INITIAL_CONTROL_STATE, MachineState, ControlState, SystemState, FaultLevel, FAULT_CODES, ConnectionConfig, WorkMode, DiagnosisResult, DiagnosisLabel } from './types';
import { generateControlPacket } from './services/canProtocol';
import { wsService } from './services/websocketService';
import { LeftDataPanel } from './components/LeftDataPanel';
import { IndustrialSchematic } from './components/IndustrialSchematic';
import { BottomControlPanel } from './components/BottomControlPanel';
import { RightButtonPanel } from './components/RightButtonPanel';
import { RealTimeChart } from './components/Charts';
import { AlarmDrawer } from './components/AlarmDrawer';
import { DiagnosisPanel } from './components/DiagnosisPanel';
import { Wifi, WifiOff, Save, Square, AlertCircle } from 'lucide-react';

// File System Access API Types
interface FileSystemWritableFileStream extends WritableStream {
    write(data: any): Promise<void>;
    seek(position: number): Promise<void>;
    truncate(size: number): Promise<void>;
}

interface FileSystemFileHandle {
    createWritable(options?: any): Promise<FileSystemWritableFileStream>;
}

declare global {
    interface Window {
        showSaveFilePicker(options?: any): Promise<FileSystemFileHandle>;
    }
}

const HISTORY_LENGTH = 100;

type ViewType = 'monitor' | 'charts' | 'control' | 'alarms';

interface FaultLog {
    id: number;
    time: string;
    level: FaultLevel;
    code: number;
    description: string;
}

function App() {
    const [machine, setMachine] = useState<MachineState>(INITIAL_MACHINE_STATE);
    const [control, setControl] = useState<ControlState>(INITIAL_CONTROL_STATE);
    const [isConnected, setIsConnected] = useState(false);
    const [activeView, setActiveView] = useState<ViewType>('monitor');
    const [isAlarmDrawerOpen, setIsAlarmDrawerOpen] = useState(false);
    const [connectionConfig, setConnectionConfig] = useState<ConnectionConfig>({
        interfaceType: 'virtual',
        channel: 'can0',
        bitrate: '250000'
    });

    // Chart History
    const [history, setHistory] = useState<any[]>([]);
    // Fault History
    const [faultLogs, setFaultLogs] = useState<FaultLog[]>([]);
    // Diagnosis Result
    const [diagnosis, setDiagnosis] = useState<DiagnosisResult | null>(null);

    // Logging Refs
    const writableStreamRef = React.useRef<FileSystemWritableFileStream | null>(null);
    const bufferRef = React.useRef<string[]>([]);
    const [isLogging, setIsLogging] = useState(false);

    // 当前时间
    const [currentTime, setCurrentTime] = useState(new Date());
    useEffect(() => {
        const timer = setInterval(() => setCurrentTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

    // Logging Control
    const handleToggleLog = async () => {
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
                console.error("Error stopping log:", err);
                alert("停止记录时发生错误，部分数据可能未保存。");
            }
            return;
        }

        try {
            if (!('showSaveFilePicker' in window)) {
                alert("当前浏览器不支持本地文件写入 API (File System Access API)。请使用 Chrome 或 Edge 桌面版。");
                return;
            }

            const nowStr = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            const handle = await window.showSaveFilePicker({
                suggestedName: `HMI_Log_${nowStr}.csv`,
                types: [{
                    description: 'CSV Data Log',
                    accept: { 'text/csv': ['.csv'] },
                }],
            });

            const stream = await handle.createWritable();
            await stream.write("Timestamp,Stack_Voltage(V),Stack_Current(A),Stack_Temp(C),H2_Pressure(MPa),DCDC_Voltage(V),DCDC_Current(A),Fan1_Duty(%),Fault_Code\n");

            writableStreamRef.current = stream;
            setIsLogging(true);

        } catch (err: any) {
            if (err.name !== 'AbortError') {
                console.error("Failed to start logging:", err);
                alert("无法创建日志文件: " + err.message);
            }
        }
    };

    // Flush Buffer Interval
    useEffect(() => {
        const interval = setInterval(async () => {
            if (isLogging && writableStreamRef.current && bufferRef.current.length > 0) {
                try {
                    const chunk = bufferRef.current.join('');
                    bufferRef.current = [];
                    await writableStreamRef.current.write(chunk);
                } catch (err) {
                    console.error("Write error:", err);
                }
            }
        }, 2000);

        return () => clearInterval(interval);
    }, [isLogging]);

    // WebSocket Connection Management
    useEffect(() => {
        if (isConnected) {
            wsService.connect('ws://localhost:8765');

            const unsubscribeState = wsService.onMachineState((state) => {
                setMachine(state);
            });

            const unsubscribeConnection = wsService.onConnection((connected) => {
                if (!connected) {
                    setMachine(prev => ({ ...prev, connected: false }));
                }
            });

            const unsubscribeDiagnosis = wsService.onDiagnosis((result) => {
                setDiagnosis(result);
            });

            return () => {
                unsubscribeState();
                unsubscribeConnection();
                unsubscribeDiagnosis();
            };
        } else {
            wsService.disconnect();
            setMachine(prev => ({ ...prev, connected: false }));
        }

        return () => { };
    }, [isConnected]);

    // Update History for Charts & Fault Logs
    useEffect(() => {
        if (!isConnected) return;

        setHistory(prev => {
            const now = Date.now();
            // 最多保留10分钟数据(600秒)
            const maxSeconds = 600;

            // 先清理超过10分钟的旧数据
            let filteredHistory = prev.filter(p => (now - p.timestamp) / 1000 <= maxSeconds);

            // 计算相对于最早数据点的秒数
            const baseTime = filteredHistory.length > 0 ? filteredHistory[0].timestamp : now;

            const newPoint = {
                time: Math.round((now - baseTime) / 1000), // 相对秒数
                timestamp: now,
                voltage: machine.power.stackVoltage,
                current: machine.power.stackCurrent,
                temp: machine.sensors.stackTemp
            };

            // 重新计算所有点的相对时间
            const newHistory = [...filteredHistory, newPoint].map(p => ({
                ...p,
                time: Math.round((p.timestamp - baseTime) / 1000)
            }));

            return newHistory;
        });

        if (machine.io.faultCode !== 0) {
            setFaultLogs(prev => {
                const lastLog = prev[0];
                if (!lastLog || lastLog.code !== machine.io.faultCode || (Date.now() - new Date('1970/01/01 ' + lastLog.time).getTime() > 2000)) {
                    const newLog: FaultLog = {
                        id: Date.now(),
                        time: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
                        level: machine.status.faultLevel,
                        code: machine.io.faultCode,
                        description: FAULT_CODES[machine.io.faultCode] || "未知故障"
                    };
                    return [newLog, ...prev].slice(0, 50);
                }
                return prev;
            });
        }

    }, [machine, isConnected]);

    // Data Logging Collection (Buffered)
    useEffect(() => {
        if (!isLogging) return;

        const now = new Date().toLocaleString('zh-CN', { hour12: false });
        const line = `${now},${machine.power.stackVoltage.toFixed(1)},${machine.power.stackCurrent.toFixed(1)},${machine.sensors.stackTemp.toFixed(1)},${machine.sensors.h2InletPressure.toFixed(2)},${machine.power.dcfOutVoltage.toFixed(1)},${machine.power.dcfOutCurrent.toFixed(1)},${machine.io.fan1Duty},${machine.io.faultCode}\n`;

        bufferRef.current.push(line);
    }, [machine, isLogging]);

    // Handle Control Updates (TX)
    const handleControlUpdate = (updates: Partial<ControlState>) => {
        const newControl = { ...control, ...updates };

        const hasChanged = Object.keys(updates).some(key =>
            control[key as keyof ControlState] !== updates[key as keyof ControlState]
        );

        if (!hasChanged) {
            return;
        }

        setControl(newControl);
        wsService.sendControl(newControl);

        const packet = generateControlPacket(newControl);
        console.log("TX CAN ID:", packet.id.toString(16), "DATA:", packet.data);
    };

    const getStatusText = (state: SystemState) => {
        switch (state) {
            case SystemState.OFF: return "待机";
            case SystemState.START: return "启动中";
            case SystemState.RUN: return "运行中";
            case SystemState.FAULT: return "故障";
            default: return "未知";
        }
    };

    const getLevelColor = (level: FaultLevel) => {
        switch (level) {
            case FaultLevel.WARNING: return "text-amber-400";
            case FaultLevel.SEVERE: return "text-orange-400";
            case FaultLevel.EMERGENCY: return "text-red-400";
            default: return "text-slate-500";
        }
    };

    const getLevelText = (level: FaultLevel) => {
        switch (level) {
            case FaultLevel.WARNING: return "警告";
            case FaultLevel.SEVERE: return "严重";
            case FaultLevel.EMERGENCY: return "紧急";
            default: return "提示";
        }
    };

    // 诊断反馈处理
    const handleDiagnosisFeedback = (label: DiagnosisLabel) => {
        wsService.sendDiagnosisFeedback(label);
        console.log("发送诊断反馈:", label);
    };

    return (
        <div className="h-screen flex flex-col bg-[#050A14] text-slate-100 font-sans overflow-hidden">

            {/* 顶部标题栏 - 深空幽蓝主题 */}
            <header className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b border-slate-700/50 px-4 py-2.5 flex justify-between items-center">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <span className="text-cyan-400 text-xl">◈</span>
                        <h1 className="text-lg font-bold text-slate-100">氢燃料电池监控系统</h1>
                    </div>
                    <span className={`px-3 py-1 rounded text-xs font-bold border ${machine.status.state === SystemState.RUN ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50' :
                        machine.status.state === SystemState.FAULT ? 'bg-red-500/20 text-red-400 border-red-500/50 animate-pulse' :
                            machine.status.state === SystemState.START ? 'bg-amber-500/20 text-amber-400 border-amber-500/50' :
                                'bg-slate-700/50 text-slate-400 border-slate-600'
                        }`}>
                        {getStatusText(machine.status.state)}
                    </span>
                    {machine.status.state === SystemState.FAULT && (
                        <span className="bg-red-500/20 text-red-400 border border-red-500/50 text-xs px-2 py-1 rounded animate-pulse">
                            {FAULT_CODES[machine.io.faultCode] || `故障码: ${machine.io.faultCode}`}
                        </span>
                    )}
                </div>

                <div className="flex items-center gap-4">
                    {/* 数据记录按钮 */}
                    <button
                        onClick={handleToggleLog}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded border text-xs font-bold transition-all ${isLogging
                            ? 'bg-red-500/20 border-red-500/50 text-red-400 animate-pulse'
                            : 'bg-slate-800/50 border-slate-600 text-slate-400 hover:bg-slate-700/50 hover:text-slate-300'}`}
                    >
                        {isLogging ? <Square className="w-3 h-3 fill-current" /> : <Save className="w-3 h-3" />}
                        {isLogging ? '● 记录中' : '数据记录'}
                    </button>

                    {/* 连接按钮 */}
                    <button
                        onClick={() => setIsConnected(!isConnected)}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded border text-xs font-bold transition-all ${isConnected
                            ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400'
                            : 'bg-slate-800/50 border-slate-600 text-slate-400 hover:bg-slate-700/50 hover:text-slate-300'}`}
                    >
                        {isConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
                        {isConnected ? '● 已连接' : '未连接'}
                    </button>

                    {/* 日期时间 */}
                    <div className="text-slate-400 text-sm font-mono bg-slate-800/50 px-3 py-1 rounded border border-slate-700/50">
                        <span className="text-cyan-400">{currentTime.toLocaleDateString('zh-CN')}</span>
                        <span className="mx-2 text-slate-600">|</span>
                        <span className="text-slate-300">{currentTime.toLocaleTimeString('zh-CN', { hour12: false })}</span>
                    </div>
                </div>
            </header>

            {/* 主内容区 */}
            <main className="flex-1 flex overflow-hidden">
                {/* 左侧数据面板 */}
                <LeftDataPanel data={machine} />

                {/* 中央区域 */}
                <div className="flex-1 flex flex-col">
                    {activeView === 'monitor' && (
                        <IndustrialSchematic data={machine} />
                    )}

                    {activeView === 'charts' && (
                        <div className="flex-1 p-2 overflow-auto bg-slate-950/40 grid grid-cols-1 gap-2">
                            <RealTimeChart data={history} title="电堆电压曲线" dataKey="voltage" unit="V" color="#00F0FF" />
                            <RealTimeChart data={history} title="电堆电流曲线" dataKey="current" unit="A" color="#3B82F6" />
                            <RealTimeChart data={history} title="电堆温度曲线" dataKey="temp" unit="°C" color="#F59E0B" />
                        </div>
                    )}

                    {activeView === 'alarms' && (
                        <div className="flex-1 bg-slate-950/60 backdrop-blur border border-slate-700/50 flex flex-col overflow-hidden rounded-lg m-2">
                            <div className="bg-gradient-to-r from-red-900/50 to-red-800/30 border-b border-red-700/50 text-slate-100 text-sm font-bold px-4 py-2 flex items-center gap-2">
                                <AlertCircle className="w-4 h-4 text-red-400" />
                                报警履历
                                <span className="ml-auto bg-red-500/20 text-red-400 border border-red-500/50 text-xs px-2 py-0.5 rounded">
                                    {faultLogs.length} 条
                                </span>
                            </div>
                            <div className="flex-1 overflow-auto">
                                <table className="w-full text-xs">
                                    <thead className="bg-slate-900/80 sticky top-0">
                                        <tr className="text-slate-400">
                                            <th className="px-3 py-2 text-left border-b border-slate-700/50">时间</th>
                                            <th className="px-3 py-2 text-left border-b border-slate-700/50">等级</th>
                                            <th className="px-3 py-2 text-left border-b border-slate-700/50">代码</th>
                                            <th className="px-3 py-2 text-left border-b border-slate-700/50">说明</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {faultLogs.length === 0 ? (
                                            <tr>
                                                <td colSpan={4} className="px-4 py-8 text-center text-slate-600">
                                                    系统正常，无报警记录
                                                </td>
                                            </tr>
                                        ) : (
                                            faultLogs.map(log => (
                                                <tr key={log.id} className="hover:bg-slate-800/30 border-b border-slate-800/50">
                                                    <td className="px-3 py-2 font-mono text-slate-400">{log.time}</td>
                                                    <td className={`px-3 py-2 font-bold ${getLevelColor(log.level)}`}>{getLevelText(log.level)}</td>
                                                    <td className="px-3 py-2 font-mono text-cyan-400">0x{log.code.toString(16).toUpperCase().padStart(2, '0')}</td>
                                                    <td className="px-3 py-2 text-slate-300">{log.description}</td>
                                                </tr>
                                            ))
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {activeView === 'control' && (
                        <div className="flex-1 bg-slate-950/60 backdrop-blur border border-slate-700/50 flex flex-col overflow-hidden rounded-lg m-2 p-4">
                            <div className="bg-gradient-to-r from-slate-900 to-slate-800 border-b border-slate-700/50 text-slate-100 text-sm font-bold px-4 py-2 flex items-center gap-2 -m-4 mb-4">
                                <span className="text-cyan-400">⚙</span>
                                参数设定
                                {control.mode === WorkMode.AUTO && (
                                    <span className="ml-auto text-[10px] text-amber-400 bg-amber-500/10 border border-amber-500/30 px-2 py-0.5 rounded">
                                        自动模式下参数不可调
                                    </span>
                                )}
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 flex-1 overflow-auto">
                                {/* DCF参数设定 */}
                                <div className={`bg-slate-900/50 border border-slate-700/50 rounded-lg p-4 ${control.mode === WorkMode.AUTO ? 'opacity-50' : ''}`}>
                                    <h3 className="text-slate-100 font-bold text-sm mb-4 flex items-center gap-2">
                                        <span className="text-amber-400">⚡</span> DCF输出设定
                                    </h3>
                                    <div className="space-y-4">
                                        <div>
                                            <div className="flex justify-between text-xs text-slate-400 mb-1">
                                                <span>目标电压</span>
                                                <span className="text-cyan-300 font-mono font-bold">{control.dcfTargetVoltage.toFixed(1)} V</span>
                                            </div>
                                            <input
                                                type="range"
                                                min={0} max={60} step={0.5}
                                                value={control.dcfTargetVoltage}
                                                onChange={(e) => handleControlUpdate({ dcfTargetVoltage: parseFloat(e.target.value) })}
                                                disabled={control.mode === WorkMode.AUTO}
                                                className={`w-full h-2 bg-slate-700 rounded-lg appearance-none accent-cyan-500 ${control.mode === WorkMode.AUTO ? 'cursor-not-allowed' : 'cursor-pointer'}`}
                                            />
                                        </div>
                                        <div>
                                            <div className="flex justify-between text-xs text-slate-400 mb-1">
                                                <span>目标电流</span>
                                                <span className="text-cyan-300 font-mono font-bold">{control.dcfTargetCurrent.toFixed(1)} A</span>
                                            </div>
                                            <input
                                                type="range"
                                                min={0} max={100} step={1}
                                                value={control.dcfTargetCurrent}
                                                onChange={(e) => handleControlUpdate({ dcfTargetCurrent: parseFloat(e.target.value) })}
                                                disabled={control.mode === WorkMode.AUTO}
                                                className={`w-full h-2 bg-slate-700 rounded-lg appearance-none accent-cyan-500 ${control.mode === WorkMode.AUTO ? 'cursor-not-allowed' : 'cursor-pointer'}`}
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* 风扇设定 */}
                                <div className={`bg-slate-900/50 border border-slate-700/50 rounded-lg p-4 ${control.mode === WorkMode.AUTO ? 'opacity-50' : ''}`}>
                                    <h3 className="text-slate-100 font-bold text-sm mb-4 flex items-center gap-2">
                                        <span className="text-blue-400">⟳</span> 风扇设定
                                    </h3>
                                    <div className="space-y-4">
                                        <div>
                                            <div className="flex justify-between text-xs text-slate-400 mb-1">
                                                <span>风扇1转速</span>
                                                <span className="text-cyan-300 font-mono font-bold">{control.fan1TargetSpeed} %</span>
                                            </div>
                                            <input
                                                type="range"
                                                min={0} max={100} step={5}
                                                value={control.fan1TargetSpeed}
                                                onChange={(e) => handleControlUpdate({ fan1TargetSpeed: parseInt(e.target.value) })}
                                                disabled={control.mode === WorkMode.AUTO}
                                                className={`w-full h-2 bg-slate-700 rounded-lg appearance-none accent-blue-500 ${control.mode === WorkMode.AUTO ? 'cursor-not-allowed' : 'cursor-pointer'}`}
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* CAN通信配置 */}
                                <div className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-4 md:col-span-2">
                                    <h3 className="text-slate-100 font-bold text-sm mb-4 flex items-center gap-2">
                                        <span className="text-purple-400">📡</span> 通信配置
                                    </h3>
                                    <div className="grid grid-cols-3 gap-4">
                                        <div>
                                            <label className="text-xs text-slate-400 block mb-1">接口类型</label>
                                            <select
                                                value={connectionConfig.interfaceType}
                                                onChange={(e) => setConnectionConfig(prev => ({ ...prev, interfaceType: e.target.value }))}
                                                className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
                                            >
                                                <option value="virtual">Virtual</option>
                                                <option value="socketcan">SocketCAN</option>
                                                <option value="pcan">PCAN</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-400 block mb-1">通道</label>
                                            <input
                                                type="text"
                                                value={connectionConfig.channel}
                                                onChange={(e) => setConnectionConfig(prev => ({ ...prev, channel: e.target.value }))}
                                                className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
                                            />
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-400 block mb-1">波特率</label>
                                            <select
                                                value={connectionConfig.bitrate}
                                                onChange={(e) => setConnectionConfig(prev => ({ ...prev, bitrate: e.target.value }))}
                                                className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
                                            >
                                                <option value="125000">125 kbps</option>
                                                <option value="250000">250 kbps</option>
                                                <option value="500000">500 kbps</option>
                                                <option value="1000000">1 Mbps</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* 右侧区域 */}
                <div className="w-64 flex flex-col gap-2 p-2">
                    {/* 诊断面板 */}
                    <DiagnosisPanel
                        diagnosis={diagnosis}
                        onFeedback={handleDiagnosisFeedback}
                    />
                    {/* 导航按钮面板 */}
                    <RightButtonPanel activeView={activeView} onViewChange={(v) => setActiveView(v as ViewType)} />
                </div>
            </main>

            {/* 底部控制面板 */}
            <BottomControlPanel control={control} onUpdate={handleControlUpdate} />

            {/* 底部状态栏 */}
            <footer className="bg-slate-900/80 border-t border-slate-700/50 px-4 py-1.5 text-[10px] text-slate-500 font-mono flex justify-between">
                <span>
                    <span className="text-cyan-500">CAN Rx:</span> 0x18FF01F0, 0x18FF02F0, 0x18FF03F0, 0x18FF04F0
                    <span className="mx-2 text-slate-700">|</span>
                    <span className="text-amber-500">Tx:</span> 0x18FF10A0
                </span>
                <span>Bitrate: <span className="text-slate-400">{connectionConfig.bitrate}</span> bps</span>
            </footer>

            <AlarmDrawer
                isOpen={isAlarmDrawerOpen}
                onClose={() => setIsAlarmDrawerOpen(false)}
                logs={faultLogs}
            />
        </div>
    );
}

export default App;
