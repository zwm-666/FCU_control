
import React, { useState, useEffect } from 'react';
import { INITIAL_MACHINE_STATE, INITIAL_CONTROL_STATE, MachineState, ControlState, SystemState, FaultLevel, FAULT_CODES, ConnectionConfig } from './types';
import { generateControlPacket } from './services/canProtocol';
import { wsService } from './services/websocketService';
import { SchematicView } from './components/SchematicView';
import { Gauge } from './components/Gauge';
import { ControlPanel } from './components/ControlPanel';
import { RealTimeChart } from './components/Charts';
import { Activity, AlertTriangle, Wifi, WifiOff, LayoutDashboard, LineChart, Settings2, AlertCircle } from 'lucide-react';

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
    const [connectionConfig, setConnectionConfig] = useState<ConnectionConfig>({
        interfaceType: 'virtual',
        channel: 'can0',
        bitrate: '250000'
    });

    // Chart History
    const [history, setHistory] = useState<any[]>([]);
    // Fault History
    const [faultLogs, setFaultLogs] = useState<FaultLog[]>([]);

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
        <div className="min-h-screen bg-slate-900 text-slate-200 font-sans selection:bg-cyan-500/30 flex flex-col">

            {/* HEADER */}
            <header className="bg-slate-950 border-b border-slate-800 px-6 py-3 flex justify-between items-center shadow-md z-50 sticky top-0">
                <div className="flex items-center gap-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-cyan-500/20">
                        <Activity className="text-white w-6 h-6" />
                    </div>
                    <div className="hidden md:block">
                        <h1 className="text-lg font-bold tracking-tight text-white leading-tight">氢燃料电池监控系统</h1>
                        <p className="text-[10px] text-slate-500 font-mono">FCU-2025-X01</p>
                    </div>
                </div>

                {/* Tab Navigation */}
                <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-800">
                    {[
                        { id: 'monitor', label: '实时监控', icon: LayoutDashboard },
                        { id: 'charts', label: '数据曲线', icon: LineChart },
                        { id: 'control', label: '系统控制', icon: Settings2 },
                    ].map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveView(tab.id as ViewType)}
                            className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeView === tab.id
                                ? 'bg-slate-800 text-cyan-400 shadow-sm ring-1 ring-slate-700'
                                : 'text-slate-500 hover:text-slate-300'
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
                                    <AlertTriangle className="w-5 h-5 animate-pulse" />
                                    <span>{getStatusText(machine.status.state)}</span>
                                    <span className="bg-red-500 text-white text-[10px] px-1.5 py-0.5 rounded ml-1 animate-pulse shadow-red-500/50 shadow-sm whitespace-nowrap">
                                        {FAULT_CODES[machine.io.faultCode] || `代码: ${machine.io.faultCode}`}
                                    </span>
                                </div>
                            ) : (
                                getStatusText(machine.status.state)
                            )}
                        </div>
                    </div>

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
            </header>

            {/* CONTENT AREA */}
            <main className="flex-1 overflow-auto p-6 relative">

                {/* VIEW: MONITOR */}
                {activeView === 'monitor' && (
                    <div className="space-y-4 max-w-6xl mx-auto animate-in fade-in zoom-in-95 duration-300">
                        {/* 2. Schematic View (Full Width) */}
                        <div className="w-full">
                            <SchematicView data={machine} />
                        </div>

                        {/* 1. Compact Gauges Row */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <Gauge size="small" label="电堆电压" value={machine.power.stackVoltage} unit="V" max={60} color="text-yellow-400" />
                            <Gauge size="small" label="电堆电流" value={machine.power.stackCurrent} unit="A" max={50} color="text-yellow-400" />
                            <Gauge size="small" label="电堆温度" value={machine.sensors.stackTemp} unit="°C" min={-20} max={100} color="text-orange-400" />
                            <Gauge size="small" label="氢气入口压力" value={machine.sensors.h2InletPressure} unit="MPa" max={2.5} color="text-cyan-400" />
                        </div>

                        {/* 3. Additional Mini Data (Cards) */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <div className="bg-slate-800/50 rounded border border-slate-700 p-2 flex justify-between items-center">
                                <span className="text-xs text-slate-500">DCF 输出功率</span>
                                <span className="font-mono text-sm text-slate-300">{machine.power.dcfPower} W</span>
                            </div>
                            <div className="bg-slate-800/50 rounded border border-slate-700 p-2 flex justify-between items-center">
                                <span className="text-xs text-slate-500">DCF 温度</span>
                                <span className="font-mono text-sm text-slate-300">{machine.io.dcfMosTemp} °C</span>
                            </div>
                            <div className="bg-slate-800/50 rounded border border-slate-700 p-2 flex justify-between items-center">
                                <span className="text-xs text-slate-500">风扇1 占空比</span>
                                <span className="font-mono text-sm text-slate-300">{machine.io.fan1Duty} %</span>
                            </div>
                            <div className="bg-slate-800/50 rounded border border-slate-700 p-2 flex justify-between items-center">
                                <span className="text-xs text-slate-500">氢气浓度</span>
                                <span className="font-mono text-sm text-slate-300">{machine.sensors.h2Concentration} %</span>
                            </div>
                        </div>

                        {/* 4. Fault Table (Bottom, Full Width) */}
                        <div className="bg-slate-800 rounded-xl border border-slate-700 flex flex-col overflow-hidden h-[250px] shadow-lg">
                            <div className="bg-slate-900/50 p-3 border-b border-slate-700 flex items-center justify-between">
                                <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                    <AlertCircle className="w-4 h-4 text-red-400" /> 报警信息
                                </h3>
                                <span className="text-[10px] text-slate-500 bg-slate-950 px-2 py-0.5 rounded-full">{faultLogs.length} 条记录</span>
                            </div>
                            <div className="flex-1 overflow-auto p-0 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-slate-900">
                                <table className="w-full text-xs text-left">
                                    <thead className="text-slate-500 bg-slate-900 sticky top-0 font-medium z-10">
                                        <tr>
                                            <th className="px-3 py-2 bg-slate-900">时间</th>
                                            <th className="px-3 py-2 bg-slate-900">等级</th>
                                            <th className="px-3 py-2 bg-slate-900">代码</th>
                                            <th className="px-3 py-2 bg-slate-900">说明</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-700/50">
                                        {faultLogs.length === 0 ? (
                                            <tr>
                                                <td colSpan={4} className="px-4 py-8 text-center text-slate-600 italic">
                                                    系统正常，无报警记录
                                                </td>
                                            </tr>
                                        ) : (
                                            faultLogs.map(log => (
                                                <tr key={log.id} className="hover:bg-slate-700/30 transition-colors">
                                                    <td className="px-3 py-2 font-mono text-slate-400">{log.time}</td>
                                                    <td className={`px-3 py-2 font-bold ${getLevelColor(log.level)}`}>{getLevelText(log.level)}</td>
                                                    <td className="px-3 py-2 font-mono text-slate-500">0x{log.code.toString(16).toUpperCase().padStart(2, '0')}</td>
                                                    <td className="px-3 py-2 text-slate-300 w-1/2">{log.description}</td>
                                                </tr>
                                            ))
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}

                {/* VIEW: CHARTS */}
                {activeView === 'charts' && (
                    <div className="grid grid-cols-1 gap-4 h-full max-w-6xl mx-auto animate-in slide-in-from-right-4 duration-300">
                        <RealTimeChart data={history} title="电堆电压曲线" dataKey="voltage" unit="V" color="#facc15" />
                        <RealTimeChart data={history} title="电堆电流曲线" dataKey="current" unit="A" color="#38bdf8" />
                        <RealTimeChart data={history} title="电堆温度曲线" dataKey="temp" unit="°C" color="#fb923c" />
                    </div>
                )}

                {/* VIEW: CONTROL */}
                {activeView === 'control' && (
                    <div className="max-w-5xl mx-auto animate-in slide-in-from-right-4 duration-300">
                        <ControlPanel
                            control={control}
                            onUpdate={handleControlUpdate}
                            connectionConfig={connectionConfig}
                            onConfigUpdate={(update) => setConnectionConfig(prev => ({ ...prev, ...update }))}
                        />
                    </div>
                )}

            </main>

            {/* FOOTER */}
            <div className="bg-slate-950 border-t border-slate-800 py-1 px-4 text-center text-[10px] text-slate-600 font-mono">
                CAN Rx: 0x18FF01F0,  0x18FF02F0, 0x18FF03F0, 0x18FF04F0 | Tx: 0x18FF10A0 | Bitrate: {connectionConfig.bitrate} bps
            </div>

        </div>
    );
}

export default App;
