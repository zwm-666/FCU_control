import * as React from 'react';
import { AlertTriangle } from 'lucide-react';
import { FaultLevel, FaultLog } from '../../types';

interface Props {
  faultLogs: FaultLog[];
}

const LEVEL_STYLES: Record<FaultLevel, string> = {
  [FaultLevel.EMERGENCY]: 'bg-rose-500/20 text-rose-400 border-rose-500/50',
  [FaultLevel.SEVERE]:    'bg-orange-500/20 text-orange-400 border-orange-500/50',
  [FaultLevel.WARNING]:   'bg-amber-500/20 text-amber-400 border-amber-500/50',
  [FaultLevel.NORMAL]:    'bg-slate-700/60 text-slate-300 border-slate-600',
};

const LEVEL_TEXT: Record<FaultLevel, string> = {
  [FaultLevel.EMERGENCY]: '紧急',
  [FaultLevel.SEVERE]:    '严重',
  [FaultLevel.WARNING]:   '警告',
  [FaultLevel.NORMAL]:    '提示',
};

export const AlarmsView: React.FC<Props> = ({ faultLogs }) => (
  <div className="h-full overflow-hidden flex flex-col bg-slate-900/50 pt-14">
    {/* Header */}
    <div className="p-4 border-b border-white/10 bg-rose-900/20 shrink-0">
      <h3 className="text-lg font-bold text-rose-400 flex items-center gap-2">
        <AlertTriangle className="w-5 h-5" />
        即时报警列表
        <span className="ml-1 px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-300 text-sm font-mono">
          {faultLogs.length}
        </span>
      </h3>
    </div>

    {/* Table */}
    <div className="flex-1 overflow-auto custom-scrollbar">
      {faultLogs.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-full gap-4 text-slate-500">
          <AlertTriangle className="w-12 h-12 opacity-20" />
          <span className="text-sm">暂无报警记录</span>
        </div>
      ) : (
        <table className="w-full text-sm text-left">
          <thead className="text-xs text-slate-400 uppercase bg-black/20 sticky top-0 backdrop-blur z-10">
            <tr>
              <th className="px-6 py-3 font-semibold tracking-wider">时间</th>
              <th className="px-6 py-3 font-semibold tracking-wider">等级</th>
              <th className="px-6 py-3 font-semibold tracking-wider">代码</th>
              <th className="px-6 py-3 font-semibold tracking-wider">详细描述</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {faultLogs.map((log) => {
              const levelStyle =
                LEVEL_STYLES[log.level] ?? LEVEL_STYLES[FaultLevel.NORMAL];
              const levelText =
                LEVEL_TEXT[log.level] ?? '提示';

              return (
                <tr
                  key={log.id}
                  className="hover:bg-white/5 transition-colors"
                >
                  <td className="px-6 py-4 font-mono text-slate-300 whitespace-nowrap">
                    {log.time}
                  </td>
                  <td className="px-6 py-4">
                    <span
                      className={`px-2 py-1 rounded text-xs font-bold border ${levelStyle}`}
                    >
                      {levelText}
                    </span>
                  </td>
                  <td className="px-6 py-4 font-mono text-cyan-500 whitespace-nowrap">
                    0x{log.code.toString(16).toUpperCase().padStart(2, '0')}
                  </td>
                  <td className="px-6 py-4 text-slate-200">
                    {log.description}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  </div>
);
