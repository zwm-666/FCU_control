
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
        <div className={`relative flex flex-col justify-between rounded-lg border bg-gradient-to-br p-4 transition-all hover:bg-white/5 ${activeColor} border-white/10 bg-slate-800/40`}>
            <div className="flex items-start justify-between">
                <span className="text-xs font-medium text-slate-300 uppercase tracking-wider">{label}</span>
                {icon && <div className={`opacity-80 ${color.replace('text-', 'text-')}`}>{icon}</div>}
            </div>

            <div className="mt-2 flex items-baseline gap-1">
                <span className={`text-2xl font-bold tracking-tight text-white`}>
                    {value}
                </span>
                {unit && <span className="text-sm font-medium text-slate-400">{unit}</span>}
            </div>

            {subValue && (
                <div className="mt-1 text-xs text-slate-400">
                    {subValue}
                </div>
            )}
        </div>
    );
}
