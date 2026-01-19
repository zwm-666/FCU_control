
import React, { useState, useEffect } from 'react';
import { INITIAL_MACHINE_STATE, INITIAL_CONTROL_STATE, MachineState, ControlState, SystemState, FaultLevel, FAULT_CODES, ConnectionConfig, WorkMode, DiagnosisResult, DiagnosisLabel, ControlCommand } from './types';
import { generateControlPacket } from './services/canProtocol';
import { wsService } from './services/websocketService';
import { IndustrialSchematic } from './components/IndustrialSchematic';
import { RealTimeChart } from './components/Charts';
import { AlarmDrawer } from './components/AlarmDrawer';
import { DiagnosisPanel } from './components/DiagnosisPanel';
import { GlassPanel } from './components/GlassPanel';
import { MetricCard } from './components/MetricCard';
import { Wifi, WifiOff, Save, Square, Activity, LayoutDashboard, Settings, AlertTriangle, Zap, Thermometer, Wind, Gauge, Power, RotateCcw, Octagon } from 'lucide-react';

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
    const [isSystemRunning, setIsSystemRunning] = useState(false);
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

    // Local Settings State (for DCF)
    const [localSettings, setLocalSettings] = useState({
        dcfTargetVoltage: control.dcfTargetVoltage,
        dcfTargetCurrent: control.dcfTargetCurrent,
    });

    // Sync external control state to local setting when it changes (unless editing? simplied for now)
    useEffect(() => {
        setLocalSettings({
            dcfTargetVoltage: control.dcfTargetVoltage,
            dcfTargetCurrent: control.dcfTargetCurrent
        });
    }, [control.dcfTargetVoltage, control.dcfTargetCurrent]);

    const hasUnsavedSettings =
        localSettings.dcfTargetVoltage !== control.dcfTargetVoltage ||
        localSettings.dcfTargetCurrent !== control.dcfTargetCurrent;


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
        if (isSystemRunning) {
            // Activate connection when system is running
            if (!isConnected) setIsConnected(true);
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
            // Disconnect and reset state when system is off
            if (isConnected) setIsConnected(false);
            wsService.disconnect();
            setMachine(INITIAL_MACHINE_STATE);
        }

        return () => { };
    }, [isSystemRunning]);

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

    const handleCommand = (cmd: ControlCommand) => {
        handleControlUpdate({ command: cmd });
        setTimeout(() => handleControlUpdate({ command: ControlCommand.NONE }), 200);
    };

    // 诊断反馈处理
    const handleDiagnosisFeedback = (label: DiagnosisLabel) => {
        wsService.sendDiagnosisFeedback(label);
        console.log("发送诊断反馈:", label);
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

    return (
        <div className="h-screen w-screen flex flex-col bg-[#020617] text-slate-100 font-sans overflow-hidden relative">

            {/* Background Ambient Glow */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[80vw] h-[300px] bg-cyan-500/10 blur-[100px] rounded-full pointer-events-none" />

            {/* ================= HEADER ================= */}
            <header className="shrink-0 z-50 px-6 py-2 flex justify-between items-center bg-slate-900/80 backdrop-blur-md border-b border-white/5 relative shadow-xl">
                <div className="flex items-center gap-4">
                    <div className="relative">
                        <div className="absolute inset-0 bg-cyan-400/20 blur-lg rounded-full" />
                        <div className="relative border border-cyan-500/30 bg-black/40 backdrop-blur-md p-2 rounded-lg">
                            <span className="text-cyan-400 text-2xl font-bold">◈</span>
                        </div>
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">
                            氢燃料电池监控系统
                        </h1>
                        <div className="flex items-center gap-2 text-xs font-mono text-cyan-500/80">
                            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                            实时在线监控
                        </div>
                    </div>
                </div>

                {/* Center Navigation - Relocated View Tabs */}
                <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
                    <div className="flex bg-slate-800/50 backdrop-blur-md rounded-full p-1 border border-white/5">
                        {[
                            { id: 'monitor', icon: Activity, label: '监控' },
                            { id: 'charts', icon: Activity, label: '图表' },
                            { id: 'control', icon: Settings, label: '设置' },
                            { id: 'alarms', icon: AlertTriangle, label: '报警' },
                        ].map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveView(tab.id as ViewType)}
                                className={`px-4 py-1.5 rounded-full text-xs font-medium transition-all flex items-center gap-2 ${activeView === tab.id
                                    ? 'bg-cyan-500 text-slate-900 shadow-lg shadow-cyan-500/20'
                                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                                    }`}
                            >
                                <tab.icon className="w-3 h-3" />
                                {tab.label}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="flex items-center gap-6">
                    {/* System Status Badge */}
                    <GlassPanel className="!p-1.5 px-3 flex items-center gap-3 !bg-slate-800/50">
                        <div className={`w-2.5 h-2.5 rounded-full animate-pulse ${machine.status.state === SystemState.RUN ? 'bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.5)]' :
                            machine.status.state === SystemState.FAULT ? 'bg-rose-400 shadow-[0_0_10px_rgba(251,113,133,0.5)]' :
                                machine.status.state === SystemState.START ? 'bg-amber-400' : 'bg-slate-400'
                            }`} />
                        <span className="text-xs font-bold text-slate-200">{getStatusText(machine.status.state)}</span>
                    </GlassPanel>

                    {/* Clock */}
                    <div className="flex flex-col items-end">
                        <span className="text-xl font-mono text-cyan-400 tracking-wide font-bold leading-none">
                            {currentTime.toLocaleTimeString('zh-CN', { hour12: false })}
                        </span>
                        <span className="text-[10px] font-mono text-slate-400 tracking-wider uppercase mt-1">
                            {currentTime.toLocaleDateString('zh-CN')}
                        </span>
                    </div>

                    {/* Quick Connectivity */}
                    <div className="flex items-center gap-2 border-l border-white/10 pl-6">
                        <button
                            onClick={handleToggleLog}
                            className={`p-2 rounded-lg transition-all ${isLogging
                                ? 'bg-rose-500/20 text-rose-400 shadow-[0_0_10px_rgba(244,63,94,0.2)]'
                                : 'bg-white/5 text-slate-500 hover:text-slate-300'}`}
                            title="数据记录"
                        >
                            {isLogging ? <Square className="w-4 h-4 fill-current animate-pulse" /> : <Save className="w-4 h-4" />}
                        </button>
                        <button
                            onClick={() => {
                                if (isSystemRunning) {
                                    // 关闭时需要二次确认
                                    if (window.confirm('确认停止系统？这将断开与燃料电池的连接。')) {
                                        setIsSystemRunning(false);
                                    }
                                } else {
                                    // 开启时直接执行
                                    setIsSystemRunning(true);
                                }
                            }}
                            className={`p-2 rounded-lg transition-all ${isSystemRunning
                                ? 'bg-cyan-500/20 text-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.2)]'
                                : 'bg-white/5 text-slate-500 hover:text-slate-300'}`}
                            title={isSystemRunning ? "关闭系统" : "启动系统"}
                        >
                            {isSystemRunning ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
                        </button>
                    </div>
                </div>
            </header>

            {/* ================= MAIN GRID ================= */}
            {/* Added min-h-0 to ensure flex child can scroll internally */}
            <main className="flex-1 min-h-0 p-6 grid grid-cols-12 gap-6 relative z-0">

                {/* LEFT: VITAL STATS (3 cols) */}
                <div className="col-span-3 flex flex-col gap-4 h-full min-h-0">
                    <div className="flex-1 overflow-y-auto no-scrollbar space-y-4 pr-1">
                        <GlassPanel title="电堆核心指标" icon={<Activity className="w-3.5 h-3.5" />} className="shrink-0">
                            <div className="space-y-3">
                                <MetricCard
                                    label="电堆电压"
                                    value={machine.power.stackVoltage.toFixed(1)}
                                    unit="V"
                                    icon={<Zap className="w-4 h-4" />}
                                    color="cyan"
                                    subValue="目标值: ~48V"
                                />
                                <MetricCard
                                    label="电堆电流"
                                    value={machine.power.stackCurrent.toFixed(1)}
                                    unit="A"
                                    icon={<Zap className="w-4 h-4" />}
                                    color="blue"
                                />
                                <MetricCard
                                    label="电堆温度"
                                    value={machine.sensors.stackTemp.toFixed(1)}
                                    unit="°C"
                                    icon={<Thermometer className="w-4 h-4" />}
                                    color="amber"
                                    subValue="告警阈值: 75°C"
                                />
                            </div>
                        </GlassPanel>

                        <GlassPanel title="系统压力与DCF" icon={<Gauge className="w-3.5 h-3.5" />} className="shrink-0">
                            <div className="grid grid-cols-1 gap-4">
                                <MetricCard
                                    label="氢瓶压力"
                                    value={machine.sensors.h2CylinderPressure.toFixed(2)}
                                    unit="MPa"
                                    color="cyan"
                                />
                                <MetricCard
                                    label="进氢压力"
                                    value={machine.sensors.h2InletPressure.toFixed(2)}
                                    unit="MPa"
                                    color="emerald"
                                />
                                <MetricCard
                                    label="氢气浓度"
                                    value={machine.sensors.h2Concentration.toFixed(1)}
                                    unit="%"
                                    color={machine.sensors.h2Concentration > 1.0 ? "rose" : "blue"}
                                    subValue="告警阈值: 2.0%"
                                />
                                <div className="p-3 bg-slate-900/50 rounded-lg border border-white/5">
                                    <span className="text-xs text-slate-400 block mb-2">DCF 输出状态</span>
                                    <div className="flex justify-between items-end mb-1">
                                        <span className="text-slate-300 text-sm">输出电压</span>
                                        <span className="text-cyan-400 font-mono">{machine.power.dcfOutVoltage.toFixed(1)} V</span>
                                    </div>
                                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden mb-3">
                                        <div className="h-full bg-cyan-500" style={{ width: `${(machine.power.dcfOutVoltage / 60) * 100}%` }} />
                                    </div>
                                    <div className="flex justify-between items-end mb-1">
                                        <span className="text-slate-300 text-sm">输出电流</span>
                                        <span className="text-cyan-400 font-mono">{machine.power.dcfOutCurrent.toFixed(1)} A</span>
                                    </div>
                                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden mb-3">
                                        <div className="h-full bg-blue-500" style={{ width: `${(machine.power.dcfOutCurrent / 100) * 100}%` }} />
                                    </div>
                                    <div className="flex justify-between items-end mb-1">
                                        <span className="text-slate-300 text-sm">DCF温度</span>
                                        <span className={`font-mono ${machine.io.dcfMosTemp > 60 ? 'text-rose-400 animate-pulse' : 'text-amber-400'}`}>
                                            {machine.io.dcfMosTemp.toFixed(1)} ℃
                                        </span>
                                    </div>
                                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden mb-3">
                                        <div className={`h-full ${machine.io.dcfMosTemp > 60 ? 'bg-rose-500' : 'bg-amber-500'}`}
                                            style={{ width: `${(machine.io.dcfMosTemp / 100) * 100}%` }} />
                                    </div>
                                </div>
                            </div>
                        </GlassPanel>
                    </div>
                </div>

                {/* CENTER: DYNAMIC VIEW (6 cols) */}
                <div className="col-span-6 relative flex flex-col h-full min-h-0 shadow-2xl rounded-2xl bg-slate-900/20 border border-white/5 backdrop-blur-sm overflow-hidden group">

                    {/* View Tabs Overlay Removed (Moved to Header) */}

                    {/* Content Area */}
                    <div className="flex-1 relative overflow-auto custom-scrollbar">
                        {activeView === 'monitor' && (
                            <div className="w-full h-full p-4 flex items-center justify-center bg-transparent pt-12">
                                {/* Schematic wrapper to fit nicely */}
                                <div className="scale-95 origin-center w-full h-full flex justify-center items-center">
                                    <IndustrialSchematic data={machine} />
                                </div>
                            </div>
                        )}

                        {activeView === 'charts' && (
                            <div className="p-6 h-full flex flex-col gap-4 pt-16">
                                <h3 className="text-lg font-bold text-slate-200 mb-2">系统实时趋势分析</h3>
                                <RealTimeChart data={history} title="电堆电压趋势" dataKey="voltage" unit="V" color="#22D3EE" />
                                <RealTimeChart data={history} title="电堆电流趋势" dataKey="current" unit="A" color="#3B82F6" />
                                <RealTimeChart data={history} title="电堆温度趋势" dataKey="temp" unit="°C" color="#F59E0B" />
                            </div>
                        )}

                        {activeView === 'control' && (
                            <div className="p-8 h-full pt-16">
                                <h3 className="text-xl font-bold text-slate-100 mb-6 flex items-center gap-3">
                                    <Settings className="w-6 h-6 text-cyan-400" />
                                    系统配置中心
                                </h3>

                                {/* CAN Config */}
                                <div className="mb-8 p-6 rounded-xl border border-white/10 bg-white/5">
                                    <h4 className="text-sm font-bold text-slate-300 uppercase mb-4">通信接口配置</h4>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="text-xs text-slate-400 block mb-1">接口类型</label>
                                            <select
                                                value={connectionConfig.interfaceType}
                                                onChange={(e) => setConnectionConfig(prev => ({ ...prev, interfaceType: e.target.value }))}
                                                className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
                                            >
                                                <option value="virtual">Virtual (虚拟)</option>
                                                <option value="socketcan">SocketCAN</option>
                                                <option value="pcan">PCAN</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-400 block mb-1">通道 (Channel)</label>
                                            <input
                                                type="text"
                                                value={connectionConfig.channel}
                                                onChange={(e) => setConnectionConfig(prev => ({ ...prev, channel: e.target.value }))}
                                                className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
                                            />
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-400 block mb-1">波特率 (Bitrate)</label>
                                            <select
                                                value={connectionConfig.bitrate}
                                                onChange={(e) => setConnectionConfig(prev => ({ ...prev, bitrate: e.target.value }))}
                                                className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
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
                        )}

                        {activeView === 'alarms' && (
                            <div className="h-full overflow-hidden flex flex-col bg-slate-900/50 pt-14">
                                <div className="p-4 border-b border-white/10 bg-rose-900/20">
                                    <h3 className="text-lg font-bold text-rose-400 flex items-center gap-2">
                                        <AlertTriangle className="w-5 h-5" />
                                        即时报警列表 ({faultLogs.length})
                                    </h3>
                                </div>
                                <div className="flex-1 overflow-auto">
                                    <table className="w-full text-sm text-left">
                                        <thead className="text-xs text-slate-400 uppercase bg-black/20 sticky top-0 backdrop-blur">
                                            <tr>
                                                <th className="px-6 py-3">时间</th>
                                                <th className="px-6 py-3">等级</th>
                                                <th className="px-6 py-3">代码</th>
                                                <th className="px-6 py-3">详细描述</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-white/5">
                                            {faultLogs.map((log) => (
                                                <tr key={log.id} className="hover:bg-white/5 transition-colors">
                                                    <td className="px-6 py-4 font-mono text-slate-300">{log.time}</td>
                                                    <td className="px-6 py-4">
                                                        <span className={`px-2 py-1 rounded text-xs font-bold border ${log.level === FaultLevel.EMERGENCY ? 'bg-rose-500/20 text-rose-400 border-rose-500/50' :
                                                            log.level === FaultLevel.WARNING ? 'bg-amber-500/20 text-amber-400 border-amber-500/50' :
                                                                'bg-slate-700 text-slate-300 border-slate-600'
                                                            }`}>
                                                            {log.level === FaultLevel.WARNING ? '警告' :
                                                                log.level === FaultLevel.SEVERE ? '严重' :
                                                                    log.level === FaultLevel.EMERGENCY ? '紧急' : '提示'}
                                                        </span>
                                                    </td>
                                                    <td className="px-6 py-4 font-mono text-cyan-500">0x{log.code.toString(16).toUpperCase()}</td>
                                                    <td className="px-6 py-4 text-slate-200">{log.description}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* RIGHT: CONTROLS & DIAGNOSIS (3 cols) */}
                <div className="col-span-3 flex flex-col gap-6 h-full min-h-0 overflow-hidden">

                    {/* Control Panel: DCF & Fans */}
                    <div className="flex-1 overflow-y-auto no-scrollbar space-y-4 pr-1">
                        <GlassPanel title="手动控制面板" icon={<Settings className="w-4 h-4" />} className="shrink-0">
                            {/* Mode Toggle */}
                            <div className="flex bg-slate-900 rounded-lg p-1 border border-white/5 mb-6">
                                <button
                                    onClick={() => handleControlUpdate({ mode: WorkMode.MANUAL })}
                                    className={`flex-1 py-2 text-xs font-bold rounded flex items-center justify-center gap-2 transition-all ${control.mode === WorkMode.MANUAL ? 'bg-cyan-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-300'
                                        }`}
                                >
                                    <Settings className="w-3 h-3" /> 手动 MANUAL
                                </button>
                                <button
                                    onClick={() => handleControlUpdate({ mode: WorkMode.AUTO })}
                                    className={`flex-1 py-2 text-xs font-bold rounded flex items-center justify-center gap-2 transition-all ${control.mode === WorkMode.AUTO ? 'bg-emerald-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-300'
                                        }`}
                                >
                                    <Zap className="w-3 h-3" /> 自动 AUTO
                                </button>
                            </div>

                            {/* Warnings if Auto */}
                            {control.mode === WorkMode.AUTO && (
                                <div className="mb-4 p-3 bg-amber-500/10 border border-amber-500/20 text-amber-400 rounded text-xs flex items-center gap-2">
                                    <AlertTriangle className="w-3 h-3 shrink-0" />
                                    自动模式下禁用手动控制
                                </div>
                            )}

                            <div className={`space-y-6 ${control.mode === WorkMode.AUTO ? 'opacity-50 pointer-events-none' : ''}`}>
                                {/* DCF Controls */}
                                <div className="space-y-4">
                                    <div>
                                        <div className="flex justify-between text-xs mb-1">
                                            <span className="text-slate-300">DCF 目标电压 (V)</span>
                                            <span className="text-cyan-400 font-mono">{localSettings.dcfTargetVoltage}V</span>
                                        </div>
                                        <input
                                            type="range" min="0" max="60" step="0.5"
                                            value={localSettings.dcfTargetVoltage}
                                            onChange={(e) => setLocalSettings(prev => ({ ...prev, dcfTargetVoltage: parseFloat(e.target.value) }))}
                                            className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none accent-cyan-500 cursor-pointer"
                                        />
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-xs mb-1">
                                            <span className="text-slate-300">DCF 目标电流 (A)</span>
                                            <span className="text-cyan-400 font-mono">{localSettings.dcfTargetCurrent}A</span>
                                        </div>
                                        <input
                                            type="range" min="0" max="100" step="1"
                                            value={localSettings.dcfTargetCurrent}
                                            onChange={(e) => setLocalSettings(prev => ({ ...prev, dcfTargetCurrent: parseFloat(e.target.value) }))}
                                            className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none accent-blue-500 cursor-pointer"
                                        />
                                    </div>

                                    {/* Apply Button for DCF */}
                                    {hasUnsavedSettings && (
                                        <button
                                            onClick={() => handleControlUpdate({
                                                dcfTargetVoltage: localSettings.dcfTargetVoltage,
                                                dcfTargetCurrent: localSettings.dcfTargetCurrent
                                            })}
                                            className="w-full py-1.5 bg-cyan-500/20 border border-cyan-500/50 text-cyan-400 text-xs font-bold rounded hover:bg-cyan-500/30 transition-all"
                                        >
                                            应用设定值
                                        </button>
                                    )}
                                </div>

                                <div className="h-px bg-white/5" />

                                {/* Fan Control */}
                                <div>
                                    <div className="flex justify-between text-xs mb-1">
                                        <span className="text-slate-300">风扇 1 转速 (%)</span>
                                        <span className="text-cyan-400 font-mono">{control.fan1TargetSpeed}%</span>
                                    </div>
                                    <input
                                        type="range" min="0" max="100" step="5"
                                        value={control.fan1TargetSpeed}
                                        onChange={(e) => handleControlUpdate({ fan1TargetSpeed: parseInt(e.target.value) })}
                                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none accent-emerald-500 cursor-pointer"
                                    />
                                </div>

                                <div className="h-px bg-white/5" />

                                {/* Valve Toggles */}
                                <div className="grid grid-cols-2 gap-2">
                                    {[
                                        { label: '进气阀', key: 'forceInletValve' },
                                        { label: '排氢阀', key: 'forcePurgeValve' },
                                        { label: '加热器', key: 'forceHeater' },
                                        { label: '风扇2', key: 'forceFan2' },
                                    ].map((item) => (
                                        <button
                                            key={item.key}
                                            onClick={() => handleControlUpdate({ [item.key]: !control[item.key as keyof ControlState] })}
                                            className={`px-2 py-2 text-xs font-medium rounded border transition-all ${control[item.key as keyof ControlState]
                                                ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50 shadow-cyan-500/20'
                                                : 'bg-slate-800 text-slate-400 border-slate-700'
                                                }`}
                                        >
                                            {item.label} {control[item.key as keyof ControlState] ? 'ON' : 'OFF'}
                                        </button>
                                    ))}
                                </div>
                            </div>

                        </GlassPanel>

                        {/* System Controls (MOVED BELOW MANUAL PANEL) */}
                        <GlassPanel title="系统主控指令" icon={<Power className="w-4 h-4" />} className="shrink-0 bg-slate-900/40">
                            <div className="grid grid-cols-2 gap-3">
                                <button onClick={() => handleCommand(ControlCommand.START)} className="flex items-center justify-center gap-1.5 px-3 py-3 rounded-md bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 text-sm font-bold transition-all active:scale-95">
                                    <Power className="w-4 h-4" /> 启动系统
                                </button>
                                <button onClick={() => handleCommand(ControlCommand.SHUTDOWN)} className="flex items-center justify-center gap-1.5 px-3 py-3 rounded-md bg-slate-700/30 hover:bg-slate-700/50 text-slate-200 border border-slate-600/50 text-sm font-bold transition-all active:scale-95">
                                    <Power className="w-4 h-4" /> 停止关机
                                </button>
                                <button onClick={() => handleCommand(ControlCommand.RESET)} className="col-span-2 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30 text-xs font-bold transition-all active:scale-95">
                                    <RotateCcw className="w-3.5 h-3.5" /> 故障复位
                                </button>
                                <button onClick={() => handleCommand(ControlCommand.EMERGENCY_STOP)} className="col-span-2 flex items-center justify-center gap-1.5 px-3 py-3 rounded-md bg-rose-500/20 hover:bg-rose-500/30 text-rose-400 border border-rose-500/50 text-sm font-bold transition-all active:scale-95 shadow-[0_0_10px_rgba(244,63,94,0.2)]">
                                    <Octagon className="w-4 h-4" /> 紧急停止
                                </button>
                            </div>
                        </GlassPanel>

                        {/* Diagnosis Panel (Compact Mode) */}
                        <div className="shrink-0">
                            <DiagnosisPanel diagnosis={diagnosis} onFeedback={handleDiagnosisFeedback} />
                        </div>
                    </div>

                </div>

            </main>

            {/* ================= FOOTER ================= */}
            <footer className="h-8 bg-black/60 backdrop-blur-md border-t border-white/10 flex items-center justify-between px-6 text-[10px] text-slate-400 font-mono z-20 relative">
                <div className="flex items-center gap-4">
                    <span className="flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-500/50"></span>
                        CAN 总线通信就绪
                    </span>
                    <span className="text-slate-700">|</span>
                    <span>Bitrate: {connectionConfig.bitrate} bps</span>
                </div>

                {/* Scrolling Marquee if faulty */}
                <div className="flex-1 mx-8 relative h-full overflow-hidden flex items-center justify-center">
                    {machine.io.faultCode !== 0 && (
                        <div className="text-rose-400 font-bold animate-pulse whitespace-nowrap">
                            ⚠ 系统告警: {FAULT_CODES[machine.io.faultCode] || `未知故障代码 ${machine.io.faultCode}`} ⚠
                        </div>
                    )}
                </div>

                <div className="flex items-center gap-2">
                    <span>H2-FCU v1.2.0</span>
                    <span className="text-slate-700">|</span>
                    <span className="text-cyan-500">ANTIGRAVITY DESIGN</span>
                </div>
            </footer>

            {/* Alarm Drawer (Hidden by default, can be toggled via button if needed, or largely replaced by view) 
                Keeping it mounted though as per user requirement to not break functionality 
            */}
            <AlarmDrawer
                isOpen={isAlarmDrawerOpen}
                onClose={() => setIsAlarmDrawerOpen(false)}
                logs={faultLogs}
            />
        </div>
    );
}

export default App;
