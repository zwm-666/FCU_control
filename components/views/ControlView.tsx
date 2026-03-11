import * as React from 'react';
import { Settings } from 'lucide-react';
import { ConnectionConfig } from '../../types';

interface Props {
  connectionConfig: ConnectionConfig;
  onConfigChange: (updates: Partial<ConnectionConfig>) => void;
}

export const ControlView: React.FC<Props> = ({ connectionConfig, onConfigChange }) => (
  <div className="p-8 h-full pt-16 overflow-auto">
    <h3 className="text-xl font-bold text-slate-100 mb-6 flex items-center gap-3">
      <Settings className="w-6 h-6 text-cyan-400" />
      系统配置中心
    </h3>

    <div className="mb-8 p-6 rounded-xl border border-white/10 bg-white/5">
      <h4 className="text-sm font-bold text-slate-300 uppercase tracking-wider mb-4">
        通信接口配置
      </h4>
      <div className="grid grid-cols-2 gap-4">
        {/* Interface Type */}
        <div>
          <label className="text-xs text-slate-400 block mb-1">接口类型</label>
          <select
            value={connectionConfig.interfaceType}
            onChange={(e) => onConfigChange({ interfaceType: e.target.value })}
            className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500 transition-colors"
          >
            <option value="virtual">Virtual (虚拟)</option>
            <option value="socketcan">SocketCAN</option>
            <option value="pcan">PCAN</option>
          </select>
        </div>

        {/* Channel */}
        <div>
          <label className="text-xs text-slate-400 block mb-1">通道 (Channel)</label>
          <input
            type="text"
            value={connectionConfig.channel}
            onChange={(e) => onConfigChange({ channel: e.target.value })}
            className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500 transition-colors"
            placeholder="e.g. can0"
          />
        </div>

        {/* Bitrate */}
        <div>
          <label className="text-xs text-slate-400 block mb-1">波特率 (Bitrate)</label>
          <select
            value={connectionConfig.bitrate}
            onChange={(e) => onConfigChange({ bitrate: e.target.value })}
            className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500 transition-colors"
          >
            <option value="125000">125 kbps</option>
            <option value="250000">250 kbps</option>
            <option value="500000">500 kbps</option>
            <option value="1000000">1 Mbps</option>
          </select>
        </div>
      </div>

      {/* Config summary */}
      <div className="mt-6 p-3 rounded-lg bg-slate-900/60 border border-white/5 font-mono text-xs text-slate-400 space-y-1">
        <div>
          <span className="text-slate-500">interface </span>
          <span className="text-cyan-400">{connectionConfig.interfaceType}</span>
        </div>
        <div>
          <span className="text-slate-500">channel   </span>
          <span className="text-cyan-400">{connectionConfig.channel}</span>
        </div>
        <div>
          <span className="text-slate-500">bitrate   </span>
          <span className="text-cyan-400">{connectionConfig.bitrate} bps</span>
        </div>
      </div>
    </div>

    <div className="p-4 rounded-xl border border-amber-500/20 bg-amber-500/5 text-xs text-amber-300/80 flex items-start gap-3">
      <Settings className="w-4 h-4 shrink-0 mt-0.5 text-amber-400" />
      <span>
        修改接口配置后，需重新启动系统连接才能生效。请在断开连接的状态下进行更改。
      </span>
    </div>
  </div>
);
