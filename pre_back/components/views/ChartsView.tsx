import * as React from 'react';
import { ChartDataPoint } from '../../types';
import { RealTimeChart } from '../Charts';

interface Props {
  history: ChartDataPoint[];
}

export const ChartsView: React.FC<Props> = ({ history }) => (
  <div className="p-4 h-full flex flex-col gap-3 overflow-auto">
    <h3 className="text-lg font-bold text-slate-200 shrink-0">系统实时趋势分析</h3>
    <div className="flex-1 min-h-0 flex flex-col gap-3">
      <RealTimeChart
        data={history}
        title="电堆电压趋势"
        dataKey="voltage"
        unit="V"
        color="#22D3EE"
      />
      <RealTimeChart
        data={history}
        title="电堆电流趋势"
        dataKey="current"
        unit="A"
        color="#3B82F6"
      />
      <RealTimeChart
        data={history}
        title="电堆功率趋势"
        dataKey="stackPower"
        unit="kW"
        color="#8B5CF6"
      />
      <RealTimeChart
        data={history}
        title="电堆温度趋势"
        dataKey="temp"
        unit="°C"
        color="#F59E0B"
      />
      <RealTimeChart
        data={history}
        title="氢瓶压力趋势"
        dataKey="h2Pressure"
        unit="MPa"
        color="#10B981"
      />
    </div>
  </div>
);
