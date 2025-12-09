
import React, { useState, useEffect } from 'react';
import { INITIAL_MACHINE_STATE, INITIAL_CONTROL_STATE, MachineState, ControlState, SystemState, FaultLevel, FAULT_CODES, ConnectionConfig } from './types';
import { generateControlPacket } from './services/canProtocol';
import { wsService } from './services/websocketService';
import { SchematicView } from './components/SchematicView';
import { Gauge } from './components/Gauge';
import { ControlPanel } from './components/ControlPanel';
import { RealTimeChart } from './components/Charts';
import { AlarmDrawer } from './components/AlarmDrawer';
import { Activity, AlertTriangle, Wifi, WifiOff, LayoutDashboard, LineChart, Settings2, AlertCircle, Maximize2, Save, Square, FileText } from 'lucide-react';

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

// Amount of history points to keep for charts
const HISTORY_LENGTH = 100;

type ViewType = 'monitor' | 'charts' | 'control';

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

    // Logging Refs
    const writableStreamRef = React.useRef<FileSystemWritableFileStream | null>(null);
    const bufferRef = React.useRef<string[]>([]);
    const [isLogging, setIsLogging] = useState(false);

    // Logging Control
    const handleToggleLog = async () => {
        // STOP LOGGING
        if (isLogging) {
            try {
                if (writableStreamRef.current) {
                    // Flush remaining data
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

        // START LOGGING
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
            // Write Header
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
                    bufferRef.current = []; // Clear buffer immediately
                    await writableStreamRef.current.write(chunk);
                } catch (err) {
                    console.error("Write error:", err);
                }
            }
        }, 2000); // Flush every 2 seconds

        return () => clearInterval(interval);
    }, [isLogging]);

    // WebSocket Connection Management
    useEffect(() => {
        if (isConnected) {
            // Connect to WebSocket server
            wsService.connect('ws://localhost:8765');

            // Subscribe to machine state updates
            const unsubscribeState = wsService.onMachineState((state) => {
                setMachine(state);
            });

            // Subscribe to connection status
            const unsubscribeConnection = wsService.onConnection((connected) => {
                if (!connected) {
                    setMachine(prev => ({ ...prev, connected: false }));
                }
            });

            return () => {
                unsubscribeState();
                unsubscribeConnection();
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

        // Charts
        setHistory(prev => {
            const now = new Date().toLocaleTimeString('zh-CN', { hour12: false });
            const newPoint = {
                time: now,
                voltage: machine.power.stackVoltage,
                current: machine.power.stackCurrent,
                temp: machine.sensors.stackTemp
            };
            const newHistory = [...prev, newPoint];
            if (newHistory.length > HISTORY_LENGTH) newHistory.shift();
            return newHistory;
        });

        // Fault Logging
        if (machine.io.faultCode !== 0) {
            setFaultLogs(prev => {
                const lastLog = prev[0];
                // Avoid duplicate spamming of same fault per update tick
                if (!lastLog || lastLog.code !== machine.io.faultCode || (Date.now() - new Date('1970/01/01 ' + lastLog.time).getTime() > 2000)) {
                    const newLog: FaultLog = {
                        id: Date.now(),
                        time: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
                        level: machine.status.faultLevel,
                        code: machine.io.faultCode,
                        description: FAULT_CODES[machine.io.faultCode] || "未知故障"
                    };
                    return [newLog, ...prev].slice(0, 50); // Keep last 50 logs
                }
                return prev;
            });
        }

    }, [machine, isConnected]);

    // Data Logging Collection (Buffered)
    useEffect(() => {
        if (!isLogging) return;

        const now = new Date().toLocaleString('zh-CN', { hour12: false }); // YYYY/MM/DD HH:mm:ss
        // Format: Time, V, I, T, P, DCDC_V, DCDC_I, Fan, Code
        const line = `${now},${machine.power.stackVoltage.toFixed(1)},${machine.power.stackCurrent.toFixed(1)},${machine.sensors.stackTemp.toFixed(1)},${machine.sensors.h2InletPressure.toFixed(2)},${machine.power.dcfVoltage.toFixed(1)},${machine.power.dcfCurrent.toFixed(1)},${machine.io.fan1Duty},${machine.io.faultCode}\n`;

        bufferRef.current.push(line);
    }, [machine, isLogging]);

    // Handle Control Updates (TX)
    const handleControlUpdate = (updates: Partial<ControlState>) => {
        const newControl = { ...control, ...updates };

        // Only send if values actually changed
        const hasChanged = Object.keys(updates).some(key =>
            control[key as keyof ControlState] !== updates[key as keyof ControlState]
        );

        if (!hasChanged) {
            return; // No change, don't send
        }

        setControl(newControl);

        // Send control command via WebSocket to backend
        wsService.sendControl(newControl);

        // Log for debugging
        const packet = generateControlPacket(newControl);
        console.log("TX CAN ID:", packet.id.toString(16), "DATA:", packet.data);
    };

    const getStatusColor = (state: SystemState) => {
        switch (state) {
            case SystemState.OFF: return "text-slate-500";
            case SystemState.START: return "text-blue-400";
            case SystemState.RUN: return "text-emerald-400";
            case SystemState.FAULT: return "text-red-500";
            default: return "text-slate-500";
        }
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
            case FaultLevel.WARNING: return "text-yellow-400";
            case FaultLevel.SEVERE: return "text-orange-500";
            case FaultLevel.EMERGENCY: return "text-red-500";
            default: return "text-slate-400";
        }
    }

    const getLevelText = (level: FaultLevel) => {
        switch (level) {
            case FaultLevel.WARNING: return "警告";
            case FaultLevel.SEVERE: return "严重";
            case FaultLevel.EMERGENCY: return "紧急";
            default: return "提示";
        }
    }

    return (
        <div className="min-h-screen bg-slate-200 text-slate-800 font-sans selection:bg-cyan-500/30 flex flex-col">

            {/* HEADER */}
            <header className="bg-slate-950 border-b border-slate-800 px-6 py-3 flex justify-between items-center shadow-sm z-50 sticky top-0">
                <div className="flex items-center gap-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-cyan-600 rounded-lg flex items-center justify-center shadow-md shadow-blue-500/20">
                        <Activity className="text-white w-6 h-6" />
                    </div>
                    <div className="hidden md:block">
                        <h1 className="text-lg font-bold tracking-tight text-white leading-tight">氢燃料电池监控系统</h1>
                    </div>
                </div>

                {/* Tab Navigation */}
                <div className="flex bg-slate-200 p-1 rounded-lg border border-slate-300">
                    {[
                        { id: 'monitor', label: '实时监控', icon: LayoutDashboard },
                        { id: 'charts', label: '数据曲线', icon: LineChart },
                        { id: 'control', label: '系统控制', icon: Settings2 },
                    ].map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveView(tab.id as ViewType)}
                            className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeView === tab.id
                                ? 'bg-white text-blue-700 shadow-sm ring-1 ring-slate-300'
                                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-300/50'
                                }`}
                        >
                            <tab.icon className="w-4 h-4" />
                            {tab.label}
                        </button>
                    ))}
                </div>

                <div className="flex items-center gap-6">
                    <div className="flex flex-col items-end">
                        <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">系统状态</span>
                        <div className={`text-lg font-black tracking-tighter ${getStatusColor(machine.status.state)} flex items-center gap-2`}>
                            {/* PROMINENT FAULT DISPLAY IN HEADER */}
                            {machine.status.state === SystemState.FAULT ? (
                                <div className="flex items-center gap-2">
                                    <AlertTriangle className="w-5 h-5 animate-pulse text-red-600" />
                                    <span className="text-red-600">{getStatusText(machine.status.state)}</span>
                                    <span className="bg-red-600 text-white text-[10px] px-1.5 py-0.5 rounded ml-1 animate-pulse shadow-red-500/50 shadow-sm whitespace-nowrap">
                                        {FAULT_CODES[machine.io.faultCode] || `代码: ${machine.io.faultCode}`}
                                    </span>
                                </div>
                            ) : (
                                getStatusText(machine.status.state)
                            )}
                        </div>
                    </div>



                    <button
                        onClick={handleToggleLog}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-full border transition-all ${isLogging
                            ? 'bg-red-500/10 border-red-500/50 text-red-500 animate-pulse'
                            : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700'}`}
                        title={isLogging ? "停止记录" : "开始记录数据"}
                    >
                        {isLogging ? <Square className="w-3.5 h-3.5 fill-current" /> : <Save className="w-3.5 h-3.5" />}
                        <span className="text-xs font-bold">{isLogging ? 'REC' : '记录'}</span>
                    </button>

                    <button
                        onClick={() => setIsConnected(!isConnected)}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-full border transition-all ${isConnected
                            ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400'
                            : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700'}`}
                    >
                        {isConnected ? <Wifi className="w-3.5 h-3.5" /> : <WifiOff className="w-3.5 h-3.5" />}
                        <span className="text-xs font-bold">{isConnected ? '在线' : '离线'}</span>
                    </button>
                </div>
            </header >

            {/* CONTENT AREA */}
            < main className="flex-1 overflow-auto p-6 relative" >

                {/* VIEW: MONITOR */}
                {
                    activeView === 'monitor' && (
                        <div className="space-y-4 max-w-6xl mx-auto animate-in fade-in zoom-in-95 duration-300">
                            {/* 2. Schematic View (Full Width) */}
                            <div className="w-full">
                                <SchematicView data={machine} />
                            </div>

                            {/* 1. Compact Gauges Row */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                <Gauge size="small" label="电堆电压" value={machine.power.stackVoltage} unit="V" max={60} color="text-yellow-600" />
                                <Gauge size="small" label="电堆电流" value={machine.power.stackCurrent} unit="A" max={50} color="text-blue-600" />
                                <Gauge size="small" label="电堆温度" value={machine.sensors.stackTemp} unit="°C" min={-20} max={100} color="text-orange-600" />
                                <Gauge size="small" label="氢气入口压力" value={machine.sensors.h2InletPressure} unit="MPa" max={2.5} color="text-cyan-600" />
                            </div>

                            {/* 3. Additional Mini Data (Cards) */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                <div className="bg-slate-50 rounded border border-slate-300 p-2 flex justify-between items-center shadow-sm">
                                    <span className="text-xs text-slate-500 font-medium">DCF 输出功率</span>
                                    <span className="font-mono text-sm text-slate-700 font-bold">{machine.power.dcfPower} W</span>
                                </div>
                                <div className="bg-slate-50 rounded border border-slate-300 p-2 flex justify-between items-center shadow-sm">
                                    <span className="text-xs text-slate-500 font-medium">DCF 温度</span>
                                    <span className="font-mono text-sm text-slate-700 font-bold">{machine.io.dcfMosTemp} °C</span>
                                </div>
                                <div className="bg-slate-50 rounded border border-slate-300 p-2 flex justify-between items-center shadow-sm">
                                    <span className="text-xs text-slate-500 font-medium">风扇1 占空比</span>
                                    <span className="font-mono text-sm text-slate-700 font-bold">{machine.io.fan1Duty} %</span>
                                </div>
                                <div className="bg-slate-50 rounded border border-slate-300 p-2 flex justify-between items-center shadow-sm">
                                    <span className="text-xs text-slate-500 font-medium">氢气浓度</span>
                                    <span className="font-mono text-sm text-slate-700 font-bold">{machine.sensors.h2Concentration} %</span>
                                </div>
                            </div>

                            {/* 4. Fault Table (Bottom, Full Width) */}
                            <div className="bg-slate-50 rounded-xl border border-slate-300 flex flex-col overflow-hidden h-[250px] shadow-sm">
                                <div className="bg-slate-100 p-3 border-b border-slate-300 flex items-center justify-between">
                                    <h3 className="text-sm font-bold text-slate-700 flex items-center gap-2">
                                        <AlertCircle className="w-4 h-4 text-red-600" /> 报警信息
                                    </h3>
                                    <div className="flex items-center gap-2">
                                        <span className="text-[10px] text-slate-500 bg-white border border-slate-300 px-2 py-0.5 rounded-full shadow-sm">{faultLogs.length} 条记录</span>
                                        <button
                                            onClick={() => setIsAlarmDrawerOpen(true)}
                                            className="p-1 hover:bg-slate-200 rounded text-slate-500 transition-colors"
                                            title="展开查看全部"
                                        >
                                            <Maximize2 className="w-3.5 h-3.5" />
                                        </button>
                                    </div>
                                </div>
                                <div className="flex-1 overflow-auto p-0 scrollbar-thin scrollbar-thumb-slate-300 scrollbar-track-slate-100">
                                    <table className="w-full text-xs text-left">
                                        <thead className="text-slate-500 bg-slate-100 sticky top-0 font-medium z-10 border-b border-slate-300">
                                            <tr>
                                                <th className="px-3 py-2 bg-slate-50">时间</th>
                                                <th className="px-3 py-2 bg-slate-50">等级</th>
                                                <th className="px-3 py-2 bg-slate-50">代码</th>
                                                <th className="px-3 py-2 bg-slate-50">说明</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-slate-100">
                                            {faultLogs.length === 0 ? (
                                                <tr>
                                                    <td colSpan={4} className="px-4 py-8 text-center text-slate-400 italic">
                                                        系统正常，无报警记录
                                                    </td>
                                                </tr>
                                            ) : (
                                                faultLogs.map(log => (
                                                    <tr key={log.id} className="hover:bg-white transition-colors">
                                                        <td className="px-3 py-2 font-mono text-slate-600">{log.time}</td>
                                                        <td className={`px-3 py-2 font-bold ${getLevelColor(log.level)}`}>{getLevelText(log.level)}</td>
                                                        <td className="px-3 py-2 font-mono text-slate-500">0x{log.code.toString(16).toUpperCase().padStart(2, '0')}</td>
                                                        <td className="px-3 py-2 text-slate-700 w-1/2">{log.description}</td>
                                                    </tr>
                                                ))
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    )
                }

                {/* VIEW: CHARTS */}
                {
                    activeView === 'charts' && (
                        <div className="grid grid-cols-1 gap-4 h-full max-w-6xl mx-auto animate-in slide-in-from-right-4 duration-300">
                            <RealTimeChart data={history} title="电堆电压曲线" dataKey="voltage" unit="V" color="#ca8a04" />
                            <RealTimeChart data={history} title="电堆电流曲线" dataKey="current" unit="A" color="#0284c7" />
                            <RealTimeChart data={history} title="电堆温度曲线" dataKey="temp" unit="°C" color="#ea580c" />
                        </div>
                    )
                }

                {/* VIEW: CONTROL */}
                {
                    activeView === 'control' && (
                        <div className="max-w-5xl mx-auto animate-in slide-in-from-right-4 duration-300">
                            <ControlPanel
                                control={control}
                                onUpdate={handleControlUpdate}
                                connectionConfig={connectionConfig}
                                onConfigUpdate={(update) => setConnectionConfig(prev => ({ ...prev, ...update }))}
                            />
                        </div>
                    )
                }

            </main >

            {/* FOOTER */}
            < div className="bg-slate-50 border-t border-slate-200 py-1 px-4 text-center text-[10px] text-slate-400 font-mono" >
                CAN Rx: 0x18FF01F0, 0x18FF02F0, 0x18FF03F0, 0x18FF04F0 | Tx: 0x18FF10A0 | Bitrate: {connectionConfig.bitrate} bps
            </div >

            <AlarmDrawer
                isOpen={isAlarmDrawerOpen}
                onClose={() => setIsAlarmDrawerOpen(false)}
                logs={faultLogs}
            />

        </div >
    );
}

export default App;
