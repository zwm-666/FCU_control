import React from 'react';
import { AlertCircle, X, ChevronDown, Clock, AlertTriangle, ShieldAlert } from 'lucide-react';
import { FaultLevel } from '../types';

interface FaultLog {
    id: number;
    time: string;
    level: FaultLevel;
    code: number;
    description: string;
}

interface Props {
    isOpen: boolean;
    onClose: () => void;
    logs: FaultLog[];
}

export const AlarmDrawer: React.FC<Props> = ({ isOpen, onClose, logs }) => {

    const getLevelConfig = (level: FaultLevel) => {
        switch (level) {
            case FaultLevel.WARNING: return { color: "text-yellow-600", bg: "bg-yellow-50", border: "border-yellow-200", icon: AlertTriangle };
            case FaultLevel.SEVERE: return { color: "text-orange-600", bg: "bg-orange-50", border: "border-orange-200", icon: AlertCircle };
            case FaultLevel.EMERGENCY: return { color: "text-red-600", bg: "bg-red-50", border: "border-red-200", icon: ShieldAlert };
            default: return { color: "text-slate-600", bg: "bg-slate-50", border: "border-slate-200", icon: AlertCircle };
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
        <>
            {/* Backdrop */}
            <div
                className={`fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-40 transition-opacity duration-300 ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
                onClick={onClose}
            ></div>

            {/* Drawer */}
            <div className={`fixed inset-x-0 bottom-0 z-50 bg-white rounded-t-2xl shadow-[0_-10px_40px_rgba(0,0,0,0.1)] transition-transform duration-300 ease-out transform h-[80vh] flex flex-col ${isOpen ? 'translate-y-0' : 'translate-y-full'}`}>

                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
                    <div className="flex items-center gap-3">
                        <div className="bg-red-100 p-2 rounded-full">
                            <AlertCircle className="w-5 h-5 text-red-600" />
                        </div>
                        <div>
                            <h2 className="text-lg font-bold text-slate-800">系统报警记录</h2>
                            <p className="text-xs text-slate-500">System Alarm History</p>
                        </div>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-slate-100 rounded-full text-slate-500 transition-colors">
                        <ChevronDown className="w-6 h-6" />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-auto bg-slate-50 p-6">
                    <div className="max-w-4xl mx-auto space-y-3">
                        {logs.length === 0 ? (
                            <div className="text-center py-20 text-slate-400 flex flex-col items-center gap-4">
                                <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center">
                                    <ShieldAlert className="w-8 h-8 text-slate-300" />
                                </div>
                                <p>当前无历史报警记录</p>
                            </div>
                        ) : (
                            logs.map(log => {
                                const config = getLevelConfig(log.level);
                                const Icon = config.icon;
                                return (
                                    <div key={log.id} className={`flex items-center p-4 bg-white rounded-xl border ${config.border} shadow-sm hover:shadow-md transition-shadow`}>
                                        <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 mr-4 ${config.bg}`}>
                                            <Icon className={`w-5 h-5 ${config.color}`} />
                                        </div>

                                        <div className="flex-1 grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
                                            <div className="md:col-span-2 flex items-center gap-2 text-slate-500 font-mono text-sm">
                                                <Clock className="w-3.5 h-3.5" />
                                                {log.time}
                                            </div>

                                            <div className="md:col-span-2">
                                                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold ${config.bg} ${config.color} border ${config.border}`}>
                                                    {getLevelText(log.level)}
                                                </span>
                                            </div>

                                            <div className="md:col-span-2 font-mono text-slate-400 text-sm">
                                                Code: 0x{log.code.toString(16).toUpperCase().padStart(2, '0')}
                                            </div>

                                            <div className="md:col-span-6 font-bold text-slate-700">
                                                {log.description}
                                            </div>
                                        </div>
                                    </div>
                                );
                            })
                        )}
                    </div>
                </div>
            </div>
        </>
    );
};
