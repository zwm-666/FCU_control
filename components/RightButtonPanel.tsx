
import React from 'react';

interface Props {
    activeView: string;
    onViewChange: (view: string) => void;
}

const NavButton: React.FC<{
    label: string;
    active: boolean;
    onClick: () => void;
}> = ({ label, active, onClick }) => (
    <button
        onClick={onClick}
        className={`w-full px-2 py-2.5 text-xs font-bold border rounded transition-all ${active
            ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50 shadow-lg shadow-cyan-500/10'
            : 'bg-slate-900/50 text-slate-400 border-slate-700/50 hover:bg-slate-800/50 hover:text-slate-300'
            }`}
    >
        {label}
    </button>
);

export const RightButtonPanel: React.FC<Props> = ({ activeView, onViewChange }) => {
    return (
        <div className="w-28 bg-slate-950/80 backdrop-blur border-l border-slate-700/50 flex flex-col p-2 gap-2">
            <NavButton
                label="◈ 主界面"
                active={activeView === 'monitor'}
                onClick={() => onViewChange('monitor')}
            />
            <NavButton
                label="📈 数据曲线"
                active={activeView === 'charts'}
                onClick={() => onViewChange('charts')}
            />
            <NavButton
                label="⚙ 参数设定"
                active={activeView === 'control'}
                onClick={() => onViewChange('control')}
            />
            <NavButton
                label="⚠ 报警履历"
                active={activeView === 'alarms'}
                onClick={() => onViewChange('alarms')}
            />
        </div>
    );
};
