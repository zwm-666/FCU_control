import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Label } from 'recharts';

interface ChartProps {
  data: any[];
  title: string;
  dataKey: string;
  color: string;
  unit: string;
}

export const RealTimeChart: React.FC<ChartProps> = ({ data, title, dataKey, color, unit }) => {
  // 根据title确定Y轴标签
  const getYAxisLabel = () => {
    if (title.includes('电压')) return '电压/V';
    if (title.includes('电流')) return '电流/A';
    if (title.includes('温度')) return '温度/℃';
    return `${title}/${unit}`;
  };

  // 计算X轴刻度 - 固定间隔
  const getXAxisTicks = () => {
    if (data.length === 0) return [0];
    const maxTime = Math.max(...data.map(d => d.time || 0));
    const ticks = [];
    // 每30秒一个刻度
    const interval = 30;
    for (let i = 0; i <= maxTime; i += interval) {
      ticks.push(i);
    }
    // 确保包含最后一个值
    if (ticks[ticks.length - 1] < maxTime) {
      ticks.push(Math.ceil(maxTime / interval) * interval);
    }
    return ticks;
  };

  return (
    <div className="bg-slate-950/60 backdrop-blur border border-slate-700/50 rounded-lg p-4 h-64 flex flex-col">
      <h3 className="text-slate-100 text-xs font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
        <span className="text-cyan-400">◈</span>
        {title}
      </h3>
      <div className="flex-1 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 25 }}>
            <defs>
              <linearGradient id={`color${dataKey}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={color} stopOpacity={0.4} />
                <stop offset="95%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="time"
              type="number"
              domain={[0, 'dataMax']}
              ticks={getXAxisTicks()}
              tick={{ fill: '#94A3B8', fontSize: 10 }}
              tickLine={{ stroke: '#64748B' }}
              axisLine={{ stroke: '#64748B' }}
              allowDecimals={false}
            >
              <Label
                value="时间/s"
                position="insideBottomRight"
                offset={-5}
                fill="#94A3B8"
                fontSize={11}
              />
            </XAxis>
            <YAxis
              tick={{ fill: '#94A3B8', fontSize: 10 }}
              tickLine={{ stroke: '#64748B' }}
              axisLine={{ stroke: '#64748B' }}
              width={55}
            >
              <Label
                value={getYAxisLabel()}
                angle={-90}
                position="insideLeft"
                fill="#94A3B8"
                fontSize={11}
                style={{ textAnchor: 'middle' }}
              />
            </YAxis>
            <Tooltip
              contentStyle={{
                backgroundColor: '#1E293B',
                borderColor: '#475569',
                color: '#F1F5F9',
                borderRadius: '8px',
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5)'
              }}
              itemStyle={{ color: color }}
              formatter={(value: number) => [`${typeof value === 'number' ? value.toFixed(2) : value} ${unit}`, title]}
              labelFormatter={(label) => `时间: ${label}s`}
            />
            <Area
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              fillOpacity={1}
              fill={`url(#color${dataKey})`}
              isAnimationActive={false}
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
