
import React, { useState } from 'react';
import { ControlState, WorkMode, ControlCommand } from '../types';

interface Props {
    control: ControlState;
    onUpdate: (updates: Partial<ControlState>) => void;
}

// 确认弹窗组件
const ConfirmDialog: React.FC<{
    isOpen: boolean;
    title: string;
    message: string;
    type: 'normal' | 'warning' | 'danger';
    onConfirm: () => void;
    onCancel: () => void;
}> = ({ isOpen, title, message, type, onConfirm, onCancel }) => {
    if (!isOpen) return null;

    const typeStyles = {
        normal: 'border-cyan-500/50 bg-cyan-500/10',
        warning: 'border-amber-500/50 bg-amber-500/10',
        danger: 'border-red-500/50 bg-red-500/10',
    };

    const buttonStyles = {
        normal: 'bg-cyan-500 hover:bg-cyan-600',
        warning: 'bg-amber-500 hover:bg-amber-600',
        danger: 'bg-red-500 hover:bg-red-600',
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className={`bg-slate-900 border-2 ${typeStyles[type]} rounded-lg p-6 max-w-md mx-4 shadow-2xl`}>
                <h3 className="text-lg font-bold text-slate-100 mb-2">{title}</h3>
                <p className="text-sm text-slate-400 mb-6">{message}</p>
                <div className="flex gap-3 justify-end">
                    <button
                        onClick={onCancel}
                        className="px-4 py-2 text-sm font-bold text-slate-400 bg-slate-800 hover:bg-slate-700 border border-slate-600 rounded transition-all"
                    >
                        取消
                    </button>
                    <button
                        onClick={onConfirm}
                        className={`px-4 py-2 text-sm font-bold text-white ${buttonStyles[type]} rounded transition-all`}
                    >
                        确认
                    </button>
                </div>
            </div>
        </div>
    );
};

// 按钮组件 - 深空幽蓝主题
const ControlButton: React.FC<{
    label: string;
    onClick: () => void;
    color?: 'cyan' | 'red' | 'amber' | 'blue' | 'slate';
    active?: boolean;
    disabled?: boolean;
}> = ({ label, onClick, color = 'slate', active = false, disabled = false }) => {
    const colorClasses = {
        cyan: 'bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 border-cyan-500/50 shadow-cyan-500/20',
        red: 'bg-red-500/20 hover:bg-red-500/30 text-red-400 border-red-500/50 shadow-red-500/20',
        amber: 'bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 border-amber-500/50 shadow-amber-500/20',
        blue: 'bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 border-blue-500/50 shadow-blue-500/20',
        slate: 'bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 border-slate-600/50',
    };

    return (
        <button
            onClick={onClick}
            disabled={disabled}
            className={`px-4 py-2 text-xs font-bold border rounded transition-all ${colorClasses[color]} 
                ${active ? 'ring-2 ring-cyan-400/50 shadow-lg' : ''} 
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'active:scale-95'}`}
        >
            {label}
        </button>
    );
};

// 开关组件
const ToggleSwitch: React.FC<{
    label: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
    disabled?: boolean;
}> = ({ label, checked, onChange, disabled = false }) => (
    <div className={`flex items-center gap-2 bg-slate-900/50 border border-slate-700/50 px-3 py-1.5 rounded ${disabled ? 'opacity-40' : ''}`}>
        <span className="text-xs text-slate-400 flex-1">{label}</span>
        <button
            onClick={() => !disabled && onChange(!checked)}
            disabled={disabled}
            className={`w-12 h-6 rounded-full border flex items-center transition-all ${checked ? 'bg-cyan-500/30 border-cyan-500/50' : 'bg-slate-800 border-slate-600'
                } ${disabled ? 'cursor-not-allowed' : ''}`}
        >
            <div className={`w-5 h-5 rounded-full shadow transition-transform ${checked ? 'translate-x-6 bg-cyan-400 shadow-cyan-400/50' : 'translate-x-0.5 bg-slate-500'
                }`} />
        </button>
        <span className={`text-[10px] font-bold w-8 ${checked ? 'text-cyan-400' : 'text-slate-600'}`}>
            {checked ? 'ON' : 'OFF'}
        </span>
    </div>
);

// 数值设定组件
const ValueSetter: React.FC<{
    label: string;
    value: number;
    unit: string;
    min: number;
    max: number;
    step: number;
    onChange: (value: number) => void;
    disabled?: boolean;
}> = ({ label, value, unit, min, max, step, onChange, disabled = false }) => (
    <div className={`flex items-center gap-2 bg-slate-900/50 border border-slate-700/50 px-3 py-1.5 rounded ${disabled ? 'opacity-40' : ''}`}>
        <span className="text-xs text-slate-400">{label}</span>
        <input
            type="number"
            value={value}
            min={min}
            max={max}
            step={step}
            onChange={(e) => !disabled && onChange(parseFloat(e.target.value) || 0)}
            disabled={disabled}
            className={`w-16 px-2 py-1 text-xs font-mono text-right bg-slate-800 border border-slate-600 rounded text-cyan-300 focus:border-cyan-500 focus:outline-none ${disabled ? 'cursor-not-allowed' : ''}`}
        />
        <span className="text-xs text-slate-500">{unit}</span>
    </div>
);

// 模式选择器
const ModeSelector: React.FC<{
    mode: WorkMode;
    onChange: (mode: WorkMode) => void;
}> = ({ mode, onChange }) => (
    <div className="flex items-center gap-2 bg-slate-900/50 border border-slate-700/50 px-3 py-1.5 rounded">
        <span className="text-xs text-slate-400 font-bold">工作模式</span>
        <button
            onClick={() => onChange(WorkMode.MANUAL)}
            className={`px-3 py-1 text-xs rounded border transition-all ${mode === WorkMode.MANUAL
                ? 'bg-blue-500/30 text-blue-400 border-blue-500/50'
                : 'bg-slate-800 text-slate-500 border-slate-600 hover:bg-slate-700'
                }`}
        >
            ◉ 手动
        </button>
        <button
            onClick={() => onChange(WorkMode.AUTO)}
            className={`px-3 py-1 text-xs rounded border transition-all ${mode === WorkMode.AUTO
                ? 'bg-cyan-500/30 text-cyan-400 border-cyan-500/50'
                : 'bg-slate-800 text-slate-500 border-slate-600 hover:bg-slate-700'
                }`}
        >
            ◉ 自动
        </button>
    </div>
);

export const BottomControlPanel: React.FC<Props> = ({ control, onUpdate }) => {
    const isManual = control.mode === WorkMode.MANUAL;

    // 本地暂存的设定值（修改后未确认）
    const [localSettings, setLocalSettings] = useState({
        fan1TargetSpeed: control.fan1TargetSpeed,
        dcfTargetVoltage: control.dcfTargetVoltage,
        dcfTargetCurrent: control.dcfTargetCurrent,
    });

    // 检查是否有未确认的修改
    const hasUnsavedChanges =
        localSettings.fan1TargetSpeed !== control.fan1TargetSpeed ||
        localSettings.dcfTargetVoltage !== control.dcfTargetVoltage ||
        localSettings.dcfTargetCurrent !== control.dcfTargetCurrent;

    // 同步控制状态变化到本地
    React.useEffect(() => {
        setLocalSettings({
            fan1TargetSpeed: control.fan1TargetSpeed,
            dcfTargetVoltage: control.dcfTargetVoltage,
            dcfTargetCurrent: control.dcfTargetCurrent,
        });
    }, [control.fan1TargetSpeed, control.dcfTargetVoltage, control.dcfTargetCurrent]);

    // 确认弹窗状态
    const [confirmDialog, setConfirmDialog] = useState<{
        isOpen: boolean;
        title: string;
        message: string;
        type: 'normal' | 'warning' | 'danger';
        action: () => void;
    }>({
        isOpen: false,
        title: '',
        message: '',
        type: 'normal',
        action: () => { },
    });

    // 打开确认弹窗
    const openConfirm = (title: string, message: string, type: 'normal' | 'warning' | 'danger', action: () => void) => {
        setConfirmDialog({ isOpen: true, title, message, type, action });
    };

    // 确认操作
    const handleConfirm = () => {
        confirmDialog.action();
        setConfirmDialog(prev => ({ ...prev, isOpen: false }));
    };

    // 取消操作
    const handleCancel = () => {
        setConfirmDialog(prev => ({ ...prev, isOpen: false }));
    };

    // 状态指令处理（带二次确认）
    const handleCommand = (cmd: ControlCommand) => {
        let title = '';
        let message = '';
        let type: 'normal' | 'warning' | 'danger' = 'normal';

        switch (cmd) {
            case ControlCommand.START:
                title = '系统启动确认';
                message = '确定要启动燃料电池系统吗？请确保所有安全检查已完成。';
                type = 'warning';
                break;
            case ControlCommand.SHUTDOWN:
                title = '系统关机确认';
                message = '确定要关闭系统吗？系统将进入关机流程。';
                type = 'normal';
                break;
            case ControlCommand.RESET:
                title = '系统复位确认';
                message = '确定要复位系统吗？这将清除当前的故障状态。';
                type = 'warning';
                break;
            case ControlCommand.EMERGENCY_STOP:
                title = '⚠️ 紧急停止确认';
                message = '确定要执行紧急停止吗？这将立即切断系统所有输出！';
                type = 'danger';
                break;
        }

        openConfirm(title, message, type, () => {
            onUpdate({ command: cmd });
            // 短暂延时后清除命令
            setTimeout(() => onUpdate({ command: ControlCommand.NONE }), 200);
        });
    };

    // 确认设定值
    const handleConfirmSettings = () => {
        openConfirm(
            '确认设定值',
            `风扇转速: ${localSettings.fan1TargetSpeed}%\nDCF电压: ${localSettings.dcfTargetVoltage}V\nDCF电流: ${localSettings.dcfTargetCurrent}A\n\n确定要应用这些设定吗？`,
            'normal',
            () => {
                onUpdate({
                    fan1TargetSpeed: localSettings.fan1TargetSpeed,
                    dcfTargetVoltage: localSettings.dcfTargetVoltage,
                    dcfTargetCurrent: localSettings.dcfTargetCurrent,
                });
            }
        );
    };

    // 取消设定值修改
    const handleResetSettings = () => {
        setLocalSettings({
            fan1TargetSpeed: control.fan1TargetSpeed,
            dcfTargetVoltage: control.dcfTargetVoltage,
            dcfTargetCurrent: control.dcfTargetCurrent,
        });
    };

    return (
        <>
            <div className="bg-slate-950/80 backdrop-blur border-t border-slate-700/50">
                {/* 第一行：状态指令 */}
                <div className="flex items-center gap-4 px-4 py-2.5 border-b border-slate-800/50">
                    <div className="flex items-center gap-2 bg-slate-900/30 border border-slate-700/30 px-3 py-1.5 rounded">
                        <span className="text-xs text-slate-400 font-bold mr-2">状态指令</span>
                        <ControlButton
                            label="▶ 启动"
                            color="cyan"
                            onClick={() => handleCommand(ControlCommand.START)}
                            active={control.command === ControlCommand.START}
                        />
                        <ControlButton
                            label="■ 关机"
                            color="slate"
                            onClick={() => handleCommand(ControlCommand.SHUTDOWN)}
                            active={control.command === ControlCommand.SHUTDOWN}
                        />
                        <ControlButton
                            label="↻ 复位"
                            color="amber"
                            onClick={() => handleCommand(ControlCommand.RESET)}
                            active={control.command === ControlCommand.RESET}
                        />
                        <ControlButton
                            label="⚠ 急停"
                            color="red"
                            onClick={() => handleCommand(ControlCommand.EMERGENCY_STOP)}
                            active={control.command === ControlCommand.EMERGENCY_STOP}
                        />
                    </div>

                    <ModeSelector
                        mode={control.mode}
                        onChange={(mode) => onUpdate({ mode })}
                    />

                    {/* 设定值 - 自动模式下禁用 */}
                    <div className="flex items-center gap-2 ml-auto">
                        {!isManual && (
                            <span className="text-[10px] text-amber-400 bg-amber-500/10 border border-amber-500/30 px-2 py-0.5 rounded">
                                自动模式下设定值不可调
                            </span>
                        )}
                        <ValueSetter
                            label="风扇转速"
                            value={localSettings.fan1TargetSpeed}
                            unit="%"
                            min={0} max={100} step={5}
                            onChange={(v) => setLocalSettings(prev => ({ ...prev, fan1TargetSpeed: v }))}
                            disabled={!isManual}
                        />
                        <ValueSetter
                            label="DCF电压"
                            value={localSettings.dcfTargetVoltage}
                            unit="V"
                            min={0} max={60} step={0.5}
                            onChange={(v) => setLocalSettings(prev => ({ ...prev, dcfTargetVoltage: v }))}
                            disabled={!isManual}
                        />
                        <ValueSetter
                            label="DCF电流"
                            value={localSettings.dcfTargetCurrent}
                            unit="A"
                            min={0} max={100} step={1}
                            onChange={(v) => setLocalSettings(prev => ({ ...prev, dcfTargetCurrent: v }))}
                            disabled={!isManual}
                        />

                        {/* 确认/取消按钮 */}
                        {isManual && (
                            <div className="flex items-center gap-1 ml-2">
                                <button
                                    onClick={handleConfirmSettings}
                                    disabled={!hasUnsavedChanges}
                                    className={`px-3 py-1.5 text-xs font-bold rounded border transition-all ${hasUnsavedChanges
                                            ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50 hover:bg-cyan-500/30 animate-pulse'
                                            : 'bg-slate-800/50 text-slate-600 border-slate-700 cursor-not-allowed'
                                        }`}
                                >
                                    ✓ 确认
                                </button>
                                {hasUnsavedChanges && (
                                    <button
                                        onClick={handleResetSettings}
                                        className="px-2 py-1.5 text-xs font-bold text-slate-400 bg-slate-800/50 border border-slate-700 rounded hover:bg-slate-700/50 transition-all"
                                    >
                                        ✕
                                    </button>
                                )}
                            </div>
                        )}
                    </div>
                </div>

                {/* 第二行：手动控制 */}
                <div className={`flex items-center gap-3 px-4 py-2 ${!isManual ? 'opacity-40 pointer-events-none' : ''}`}>
                    <span className="text-xs text-slate-500 font-bold">手动控制</span>
                    <ToggleSwitch
                        label="进气阀"
                        checked={control.forceInletValve}
                        onChange={(v) => onUpdate({ forceInletValve: v })}
                        disabled={!isManual}
                    />
                    <ToggleSwitch
                        label="排氢阀"
                        checked={control.forcePurgeValve}
                        onChange={(v) => onUpdate({ forcePurgeValve: v })}
                        disabled={!isManual}
                    />
                    <ToggleSwitch
                        label="加热器"
                        checked={control.forceHeater}
                        onChange={(v) => onUpdate({ forceHeater: v })}
                        disabled={!isManual}
                    />
                    <ToggleSwitch
                        label="风扇1"
                        checked={control.forceFan1}
                        onChange={(v) => onUpdate({ forceFan1: v })}
                        disabled={!isManual}
                    />
                    <ToggleSwitch
                        label="风扇2"
                        checked={control.forceFan2}
                        onChange={(v) => onUpdate({ forceFan2: v })}
                        disabled={!isManual}
                    />
                </div>
            </div>

            {/* 确认弹窗 */}
            <ConfirmDialog
                isOpen={confirmDialog.isOpen}
                title={confirmDialog.title}
                message={confirmDialog.message}
                type={confirmDialog.type}
                onConfirm={handleConfirm}
                onCancel={handleCancel}
            />
        </>
    );
};
