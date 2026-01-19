
import React from 'react';

interface GlassPanelProps {
    children: React.ReactNode;
    className?: string;
    contentClassName?: string;
    title?: React.ReactNode;
    icon?: React.ReactNode;
    action?: React.ReactNode;
}

export function GlassPanel({ children, className = '', contentClassName = 'p-4', title, icon, action }: GlassPanelProps) {
    return (
        <div className={`relative overflow-hidden rounded-xl border border-white/10 bg-slate-900/60 backdrop-blur-md shadow-xl ${className}`}>
            {/* Glossy gradient overlay */}
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-50" />

            {/* Header if title is present */}
            {(title || icon || action) && (
                <div className="relative flex items-center justify-between border-b border-white/5 bg-white/5 px-4 py-3">
                    <div className="flex items-center gap-2 text-slate-100 font-bold tracking-wide text-sm uppercase">
                        {icon && <span className="text-cyan-400">{icon}</span>}
                        {title}
                    </div>
                    {action && <div>{action}</div>}
                </div>
            )}

            {/* Content */}
            <div className={`relative h-full ${contentClassName}`}>
                {children}
            </div>
        </div>
    );
}
