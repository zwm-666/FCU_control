
import React from 'react';
import { ControlState, WorkMode, ControlCommand, ConnectionConfig } from '../types';
import { Power, Play, RefreshCw, AlertOctagon, Settings, Fan, Flame, Cable } from 'lucide-react';

interface Props {
    control: ControlState;
    onUpdate: (updates: Partial<ControlState>) => void;
    connectionConfig: ConnectionConfig;
    onConfigUpdate: (updates: Partial<ConnectionConfig>) => void;
}

export const ControlPanel: React.FC<Props> = ({ control, onUpdate, connectionConfig, onConfigUpdate }) => {
    const isManual = control.mode === WorkMode.MANUAL;

    const handleCommand = (cmd: ControlCommand) => {
        onUpdate({ command: cmd });
        setTimeout(() => onUpdate({ command: ControlCommand.NONE }), 200);
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4">

            {/* SECTION: CONNECTION & COMMANDS */}
            <div className="md:col-span-4 flex flex-col gap-4">

                {/* COMMAND CARD */}
                <div className="bg-slate-800 rounded-lg border border-slate-700 p-4 flex flex-col gap-3 flex-1">
                    <div className="flex items-center justify-between">
                        <h3 className="text-slate-300 font-bold text-sm flex items-center gap-2">
                            <Settings className="w-4 h-4" /> 主控指令
                        </h3>
                        <div className="flex bg-slate-900 rounded p-0.5 scale-90 origin-right">
                            <button
                                onClick={() => onUpdate({ mode: WorkMode.MANUAL })}
                                className={`px-3 py-1 text-[10px] font-bold rounded transition-colors ${!isManual ? 'text-slate-500 hover:text-slate-300' : 'bg-blue-600 text-white shadow'}`}
                            >
                                手动
                            </button>
                            <button
                                onClick={() => onUpdate({ mode: WorkMode.AUTO })}
                                className={`px-3 py-1 text-[10px] font-bold rounded transition-colors ${isManual ? 'text-slate-500 hover:text-slate-300' : 'bg-emerald-600 text-white shadow'}`}
                            >
                                自动
                            </button>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 flex-1">
                        <button
                            onClick={() => handleCommand(ControlCommand.START)}
                            className="bg-emerald-500/10 border border-emerald-500/40 hover:bg-emerald-500/20 text-emerald-400 rounded flex flex-col items-center justify-center gap-1 p-2 transition-all active:scale-95 min-h-[4rem]"
                        >
                            <Play className="w-5 h-5" />
                            <span className="font-bold text-xs">启动</span>
                        </button>

                        <button
                            onClick={() => handleCommand(ControlCommand.SHUTDOWN)}
                            className="bg-slate-700/30 border border-slate-600 hover:bg-slate-700/50 text-slate-300 rounded flex flex-col items-center justify-center gap-1 p-2 transition-all active:scale-95 min-h-[4rem]"
                        >
                            <Power className="w-5 h-5" />
                            <span className="font-bold text-xs">停止</span>
                        </button>

                        <button
                            onClick={() => handleCommand(ControlCommand.RESET)}
                            className="bg-blue-500/10 border border-blue-500/40 hover:bg-blue-500/20 text-blue-400 rounded flex flex-col items-center justify-center gap-1 p-2 transition-all active:scale-95 min-h-[4rem]"
                        >
                            <RefreshCw className="w-5 h-5" />
                            <span className="font-bold text-xs">复位</span>
                        </button>

                        <button
                            onClick={() => handleCommand(ControlCommand.EMERGENCY_STOP)}
                            className="bg-red-500 hover:bg-red-600 text-white border border-red-400 rounded flex flex-col items-center justify-center gap-1 p-2 transition-all active:scale-95 shadow-[0_0_10px_rgba(239,68,68,0.4)] min-h-[4rem]"
                        >
                            <AlertOctagon className="w-5 h-5" />
                            <span className="font-bold text-xs">急停</span>
                        </button>
                    </div>
                </div>
            </div>

            {/* RIGHT: MANUAL OVERRIDES */}
            <div className={`md:col-span-8 bg-slate-800 rounded-lg border border-slate-700 p-4 transition-opacity duration-300 ${isManual ? 'opacity-100' : 'opacity-40 pointer-events-none'}`}>
                <h3 className="text-slate-300 font-bold text-sm mb-3 flex items-center gap-2">
                    <Settings className="w-4 h-4 text-orange-400" /> 手动调试参数
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Force Switches */}
                    <div className="space-y-2">
                        <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">执行器开关</h4>

                        {[
                            { label: "氢气进气阀", key: 'forceInletValve' },
                            { label: "氢气排空阀", key: 'forcePurgeValve' },
                            { label: "电堆加热膜", key: 'forceHeater', icon: Flame },
                            { label: "风扇 1 (电堆)", key: 'forceFan1', icon: Fan },
                            { label: "风扇 2 (DC/DC)", key: 'forceFan2', icon: Fan },
                        ].map((item) => (
                            <div key={item.key} className="flex items-center justify-between bg-slate-900/50 p-1.5 px-3 rounded border border-slate-700/50">
                                <div className="flex items-center gap-2 text-xs text-slate-300">
                                    {item.icon && <item.icon className="w-3.5 h-3.5 text-slate-500" />}
                                    {item.label}
                                </div>
                                <Toggle
                                    checked={(control as any)[item.key]}
                                    onChange={(v) => onUpdate({ [item.key]: v })}
                                />
                            </div>
                        ))}
                    </div>

                    {/* Sliders */}
                    <div className="space-y-4">
                        <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">模拟量设定</h4>

                        <SliderControl
                            label="风扇 1 转速"
                            value={control.fan1TargetSpeed}
                            unit="%"
                            min={0} max={100}
                            onChange={(v) => onUpdate({ fan1TargetSpeed: v })}
                        />

                        <SliderControl
                            label="DCF 目标电压"
                            value={control.dcfTargetVoltage}
                            unit="V"
                            min={0} max={65} step={0.1}
                            onChange={(v) => onUpdate({ dcfTargetVoltage: v })}
                        />

                        <SliderControl
                            label="DCF 目标电流"
                            value={control.dcfTargetCurrent}
                            unit="A"
                            min={0} max={50} step={0.1}
                            onChange={(v) => onUpdate({ dcfTargetCurrent: v })}
                        />
                    </div>
                </div>
            </div>

        </div>
    );
};

// UI Helpers
const Toggle = ({ checked, onChange }: { checked: boolean, onChange: (v: boolean) => void }) => (
    <button
        onClick={() => onChange(!checked)}
        className={`w-8 h-4 rounded-full relative transition-colors duration-200 ease-in-out flex items-center ${checked ? 'bg-blue-500' : 'bg-slate-600'}`}
    >
        <span className={`block w-3 h-3 bg-white rounded-full shadow-sm transition-transform duration-200 ${checked ? 'translate-x-4' : 'translate-x-0.5'}`} />
    </button>
);

const SliderControl = ({ label, value, unit, min, max, step = 1, onChange }: any) => (
    <div>
        <div className="flex justify-between text-[10px] text-slate-400 mb-1">
            <span>{label}</span>
            <span className="font-mono text-white">{value} {unit}</span>
        </div>
        <input
            type="range"
            min={min} max={max} step={step}
            value={value}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
    </div>
);
