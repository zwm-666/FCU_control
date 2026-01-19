
import React from 'react';

interface MetricCardProps {
    label: string;
    value: string | number;
    unit?: string;
    icon?: React.ReactNode;
    trend?: 'up' | 'down' | 'neutral';
    color?: 'cyan' | 'blue' | 'amber' | 'rose' | 'emerald';
    subValue?: string;
}

export function MetricCard({ label, value, unit, icon, color = 'cyan', subValue }: MetricCardProps) {

    const colorMap = {
        cyan: 'text-cyan-400 border-cyan-500/30 from-cyan-500/10',
        blue: 'text-blue-400 border-blue-500/30 from-blue-500/10',
        amber: 'text-amber-400 border-amber-500/30 from-amber-500/10',
        rose: 'text-rose-400 border-rose-500/30 from-rose-500/10',
        emerald: 'text-emerald-400 border-emerald-500/30 from-emerald-500/10',
    };

    const activeColor = colorMap[color];

    return (
        <div className={`relative flex flex-col justify-between rounded-xl border p-3 transition-all hover:bg-white/10 ${activeColor} border-white/10 bg-slate-800/60 shadow-lg backdrop-blur-sm group`}>
            {/* Top Shine Effect */}
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-50" />

            <div className="flex items-start justify-between mb-1">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{label}</span>
                {icon && <div className={`opacity-80 transition-transform group-hover:scale-110 ${color === 'cyan' ? 'text-cyan-400' : color === 'blue' ? 'text-blue-400' : color === 'amber' ? 'text-amber-400' : color === 'rose' ? 'text-rose-400' : 'text-emerald-400'}`}>{icon}</div>}
            </div>

            <div className="flex items-baseline gap-1">
                <span className={`text-xl font-bold tracking-tight text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.3)]`}>
                    {value}
                </span>
                {unit && <span className="text-xs font-bold text-slate-500">{unit}</span>}
            </div>

            {/* Bottom Glow */}
            <div className={`absolute inset-x-0 bottom-0 h-[2px] w-full rounded-b-xl opacity-50 bg-gradient-to-r from-transparent via-current to-transparent ${color === 'cyan' ? 'text-cyan-500' : color === 'blue' ? 'text-blue-500' : color === 'amber' ? 'text-amber-500' : color === 'rose' ? 'text-rose-500' : 'text-emerald-500'}`} />

            {subValue && (
                <div className="mt-1 text-[10px] text-slate-500 font-mono">
                    {subValue}
                </div>
            )}
        </div>
    );
}
