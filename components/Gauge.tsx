import React from 'react';

interface GaugeProps {
  label: string;
  value: number;
  unit: string;
  color?: string;
  min?: number;
  max?: number;
  size?: 'normal' | 'small';
}

export const Gauge: React.FC<GaugeProps> = ({ label, value, unit, color = "text-blue-400", min = 0, max = 100, size = 'normal' }) => {
  // Calculate percentage for a mini bar
  const percent = Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100));

  if (size === 'small') {
    return (
      <div className="bg-slate-800/80 p-2 rounded border border-slate-700/80 backdrop-blur-sm flex items-center justify-between">
        <div className="flex flex-col">
          <div className="text-slate-500 text-[10px] font-medium uppercase tracking-wider">{label}</div>
          <div className="flex items-baseline gap-1">
            <div className={`text-lg font-mono font-bold leading-none ${color}`}>
              {value}
            </div>
            <div className="text-slate-600 text-[10px]">{unit}</div>
          </div>
        </div>

        {/* Vertical Mini Bar for Small mode */}
        <div className="h-8 w-1.5 bg-slate-700 rounded-full overflow-hidden flex flex-col justify-end">
          <div
            className="w-full transition-all duration-300 ease-out opacity-80"
            style={{ height: `${percent}%`, backgroundColor: color.replace('text-', '').replace('400', '500') === color ? 'currentColor' : undefined }} // tailwind color hack usually needs style or map
          >
            {/* Inline style for bg color if mapped from props correctly or use a class map */}
            <div className={`w-full h-full ${color.replace('text-', 'bg-').replace('400', '500')}`}></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700 backdrop-blur-sm">
      <div className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-1">{label}</div>
      <div className="flex items-end justify-between">
        <div className={`text-xl font-mono font-bold ${color}`}>
          {value}
        </div>
        <div className="text-slate-500 text-sm mb-1 ml-1">{unit}</div>
      </div>
      {/* Mini Progress Bar */}
      <div className="w-full h-1.5 bg-slate-700 rounded-full mt-2 overflow-hidden">
        <div
          className="h-full bg-current opacity-80 transition-all duration-300 ease-out"
          style={{ width: `${percent}%`, color: color.replace('text-', 'bg-').replace('400', '500') }}
        >
          <div className={`w-full h-full ${color.replace('text-', 'bg-').replace('400', '500')}`}></div>
        </div>
      </div>
    </div>
  );
};