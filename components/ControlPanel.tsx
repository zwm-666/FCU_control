
import React, { useState } from 'react';
import { ControlState, WorkMode, ControlCommand, ConnectionConfig } from '../types';
import { Power, Play, RefreshCw, AlertOctagon, Settings, Fan, Flame, Cable } from 'lucide-react';
import { ConfirmationModal, ModalType } from './ConfirmationModal';

interface Props {
    control: ControlState;
    onUpdate: (updates: Partial<ControlState>) => void;
    connectionConfig: ConnectionConfig;
    onConfigUpdate: (updates: Partial<ConnectionConfig>) => void;
}

export const ControlPanel: React.FC<Props> = ({ control, onUpdate, connectionConfig, onConfigUpdate }) => {

    const isManual = control.mode === WorkMode.MANUAL;

    // Modal State
    const [modal, setModal] = useState<{
        isOpen: boolean;
        title: string;
        message: string;
        type: ModalType;
        action: () => void;
    }>({
        isOpen: false,
        title: '',
        message: '',
        type: 'info',
        action: () => { }
    });

    const openConfirm = (title: string, message: string, type: ModalType, action: () => void) => {
        setModal({ isOpen: true, title, message, type, action });
    };

    const handleConfirm = () => {
        modal.action();
        setModal(prev => ({ ...prev, isOpen: false }));
    };

    const handleCancel = () => {
        setModal(prev => ({ ...prev, isOpen: false }));
    };

    const handleCommand = (cmd: ControlCommand) => {
        let title = "确认为操作？";
        let message = "此操作将发送控制指令。";
        let type: ModalType = 'info';

        if (cmd === ControlCommand.START) {
            title = "系统启动确认";
            message = "确定要启动燃料电池系统吗？请确保所有安全检查已完成。";
            type = 'warning';
        } else if (cmd === ControlCommand.SHUTDOWN) {
            title = "系统停止确认";
            message = "确定要停止系统运行吗？系统将进入关机流程。";
            type = 'warning';
        } else if (cmd === ControlCommand.EMERGENCY_STOP) {
            title = "紧急停止确认";
            message = "确定要执行紧急停止吗？这将立即切断系统输出！";
            type = 'danger';
        } else if (cmd === ControlCommand.RESET) {
            title = "系统复位确认";
            message = "确定要复位系统状态吗？这可能会清除当前的故障状态。";
            type = 'info';
        }

        openConfirm(title, message, type, () => {
            onUpdate({ command: cmd });
            setTimeout(() => onUpdate({ command: ControlCommand.NONE }), 200);
        });
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4">

            {/* SECTION: CONNECTION & COMMANDS */}
            <div className="md:col-span-4 flex flex-col gap-4">

                {/* COMMAND CARD */}
                <div className="bg-slate-50 rounded-lg border border-slate-300 p-4 flex flex-col gap-3 flex-1 shadow-sm">
                    <div className="flex items-center justify-between">
                        <h3 className="text-slate-700 font-bold text-sm flex items-center gap-2">
                            <Settings className="w-4 h-4" /> 主控指令
                        </h3>
                        <div className="flex bg-slate-100 rounded p-0.5 scale-90 origin-right border border-slate-200">
                            <button
                                onClick={() => {
                                    if (!isManual) {
                                        openConfirm("切换至手动模式", "手动模式下系统保护可能受限，请谨慎操作。确认切换？", 'warning', () => onUpdate({ mode: WorkMode.MANUAL }));
                                    }
                                }}
                                className={`px-3 py-1 text-[10px] font-bold rounded transition-colors ${!isManual ? 'text-slate-600 hover:text-slate-800' : 'bg-blue-600 text-white shadow-sm'}`}
                            >
                                手动
                            </button>
                            <button
                                onClick={() => {
                                    if (isManual) {
                                        onUpdate({ mode: WorkMode.AUTO });
                                    }
                                }}
                                className={`px-3 py-1 text-[10px] font-bold rounded transition-colors ${isManual ? 'text-slate-600 hover:text-slate-800' : 'bg-emerald-600 text-white shadow-sm'}`}
                            >
                                自动
                            </button>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 flex-1">
                        <button
                            onClick={() => handleCommand(ControlCommand.START)}
                            className="bg-emerald-600 border border-emerald-700 hover:bg-emerald-700 text-white
                                    rounded flex flex-col items-center justify-center gap-1 p-2
                                    transition-all active:scale-95 min-h-[4rem]"
                        >
                            <Play className="w-5 h-5" />
                            <span className="font-bold text-xs">启动</span>
                        </button>


                        <button
                            onClick={() => handleCommand(ControlCommand.SHUTDOWN)}
                            className="bg-slate-50 border border-slate-200 hover:bg-slate-100 text-slate-600 rounded flex flex-col items-center justify-center gap-1 p-2 transition-all active:scale-95 min-h-[4rem]"
                        >
                            <Power className="w-5 h-5" />
                            <span className="font-bold text-xs">停止</span>
                        </button>

                        <button
                            onClick={() => handleCommand(ControlCommand.RESET)}
                            className="bg-blue-50 border border-blue-200 hover:bg-blue-100/80 text-blue-600 rounded flex flex-col items-center justify-center gap-1 p-2 transition-all active:scale-95 min-h-[4rem]"
                        >
                            <RefreshCw className="w-5 h-5" />
                            <span className="font-bold text-xs">复位</span>
                        </button>

                        <button
                            onClick={() => handleCommand(ControlCommand.EMERGENCY_STOP)}
                            className="bg-red-600 hover:bg-red-700 text-white border border-red-500 rounded flex flex-col items-center justify-center gap-1 p-2 transition-all active:scale-95 shadow-[0_0_10px_rgba(239,68,68,0.4)] min-h-[4rem]"
                        >
                            <AlertOctagon className="w-5 h-5" />
                            <span className="font-bold text-xs">急停</span>
                        </button>
                    </div>
                </div>
            </div>

            {/* RIGHT: MANUAL OVERRIDES */}
            <div className={`md:col-span-8 bg-slate-50 rounded-lg border border-slate-300 p-4 transition-opacity duration-300 shadow-sm ${isManual ? 'opacity-100' : 'opacity-40 pointer-events-none'}`}>
                <h3 className="text-slate-700 font-bold text-sm mb-3 flex items-center gap-2">
                    <Settings className="w-4 h-4 text-orange-500" /> 手动调试参数
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Force Switches */}
                    <div className="space-y-2">
                        <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">执行器开关</h4>

                        {[
                            { label: "氢气电磁阀", key: 'forceInletValve' },
                            { label: "排氢阀", key: 'forcePurgeValve' },
                            { label: "电堆加热膜", key: 'forceHeater', icon: Flame },
                            { label: "风扇 1 (电堆)", key: 'forceFan1', icon: Fan },
                            { label: "风扇 2 (DC/DC)", key: 'forceFan2', icon: Fan },
                        ].map((item) => (
                            <div key={item.key} className="flex items-center justify-between bg-white p-1.5 px-3 rounded border border-slate-300">
                                <div className="flex items-center gap-2 text-xs text-slate-700 font-medium">
                                    {item.icon && <item.icon className="w-3.5 h-3.5 text-slate-400" />}
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
                            onChange={(v: number) => openConfirm("修改风扇转速", `将风扇 1 转速设定为 ${v}%？`, 'info', () => onUpdate({ fan1TargetSpeed: v }))}
                        />

                        <SliderControl
                            label="DCF 目标电压"
                            value={control.dcfTargetVoltage}
                            unit="V"
                            min={0} max={65} step={0.1}
                            onChange={(v: number) => openConfirm("修改输出电压", `将 DCF 目标电压设定为 ${v}V？`, 'info', () => onUpdate({ dcfTargetVoltage: v }))}
                        />

                        <SliderControl
                            label="DCF 目标电流"
                            value={control.dcfTargetCurrent}
                            unit="A"
                            min={0} max={50} step={0.1}
                            onChange={(v: number) => openConfirm("修改输出电流", `将 DCF 目标电流设定为 ${v}A？`, 'info', () => onUpdate({ dcfTargetCurrent: v }))}
                        />
                    </div>
                </div>
            </div>

            <ConfirmationModal
                isOpen={modal.isOpen}
                title={modal.title}
                message={modal.message}
                type={modal.type}
                onConfirm={handleConfirm}
                onCancel={handleCancel}
            />

        </div >
    );
};

// UI Helpers
const Toggle = ({ checked, onChange }: { checked: boolean, onChange: (v: boolean) => void }) => (
    <button
        onClick={() => onChange(!checked)}
        className={`w-8 h-4 rounded-full relative transition-colors duration-200 ease-in-out flex items-center ${checked ? 'bg-blue-600' : 'bg-slate-300'}`}
    >
        <span className={`block w-3 h-3 bg-white rounded-full shadow-sm transition-transform duration-200 ${checked ? 'translate-x-4' : 'translate-x-0.5'}`} />
    </button>
);

const SliderControl = ({ label, value, unit, min, max, step = 1, onChange }: any) => {
    const [localValue, setLocalValue] = React.useState(value);

    // Sync local value with prop value
    React.useEffect(() => {
        setLocalValue(value);
    }, [value]);

    const handleCommit = () => {
        // Only send control signal when user stops dragging
        if (localValue !== value) {
            onChange(localValue);
        }
    };

    return (
        <div>
            <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                <span>{label}</span>
                <span className="font-mono text-slate-900 font-bold">{localValue} {unit}</span>
            </div>
            <input
                type="range"
                min={min} max={max} step={step}
                value={localValue}
                onChange={(e) => setLocalValue(parseFloat(e.target.value))}
                onMouseUp={handleCommit}
                onTouchEnd={handleCommit}
                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            />
        </div>
    );
};
