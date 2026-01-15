
import React from 'react';
import { MachineState } from '../types';

interface Props {
    data: MachineState;
}

// 数据显示框组件 - 深空幽蓝主题
const DataBox: React.FC<{
    label: string;
    value: number | string;
    unit?: string;
    highlight?: boolean;
}> = ({ label, value, unit = '', highlight = false }) => (
    <div className="flex justify-between items-center bg-slate-900/60 backdrop-blur border border-slate-700/50 px-2 py-1.5 rounded">
        <span className="text-xs text-slate-400">{label}</span>
        <span className={`font-mono text-sm font-bold ${highlight ? 'text-cyan-400' : 'text-slate-100'}`}>
            {typeof value === 'number' ? value.toFixed(1) : value}
            {unit && <span className="text-slate-500 text-xs ml-1">{unit}</span>}
        </span>
    </div>
);

// 状态指示灯组件
const StatusLED: React.FC<{
    label: string;
    active: boolean;
    activeColor?: string;
}> = ({ label, active, activeColor = 'bg-cyan-400 shadow-cyan-400/50' }) => (
    <div className="flex items-center gap-2 bg-slate-900/60 backdrop-blur border border-slate-700/50 px-2 py-1.5 rounded">
        <div className={`w-2.5 h-2.5 rounded-full transition-all ${active ? `${activeColor} shadow-lg` : 'bg-slate-600'}`} />
        <span className="text-xs text-slate-400">{label}</span>
        <span className={`text-[10px] font-bold ml-auto ${active ? 'text-cyan-400' : 'text-slate-600'}`}>
            {active ? 'ON' : 'OFF'}
        </span>
    </div>
);

// 分组标题组件
const SectionTitle: React.FC<{ title: string; color?: string }> = ({ title, color = 'from-blue-600 to-blue-800' }) => (
    <div className={`bg-gradient-to-r ${color} text-white text-xs font-bold px-3 py-1.5 rounded-t`}>
        {title}
    </div>
);

export const LeftDataPanel: React.FC<Props> = ({ data }) => {
    return (
        <div className="w-52 bg-slate-950/80 backdrop-blur border-r border-slate-700/50 flex flex-col text-sm overflow-auto">

            {/* 电堆参数 */}
            <div className="p-2">
                <SectionTitle title="⚡ 电堆参数" color="from-cyan-600 to-blue-700" />
                <div className="bg-slate-900/40 border border-slate-700/30 border-t-0 rounded-b p-2 space-y-1.5">
                    <DataBox label="电压" value={data.power.stackVoltage} unit="V" />
                    <DataBox label="电流" value={data.power.stackCurrent} unit="A" />
                    <DataBox label="功率" value={data.power.stackPower} unit="kW" highlight />
                </div>
            </div>

            {/* DCF输出 */}
            <div className="p-2 pt-0">
                <SectionTitle title="🔋 DCF输出" color="from-amber-600 to-orange-700" />
                <div className="bg-slate-900/40 border border-slate-700/30 border-t-0 rounded-b p-2 space-y-1.5">
                    <DataBox label="输出电压" value={data.power.dcfOutVoltage} unit="V" />
                    <DataBox label="输出电流" value={data.power.dcfOutCurrent} unit="A" />
                    <DataBox label="输出功率" value={data.power.dcfPower} unit="W" />
                    <DataBox label="效率" value={data.power.dcfEfficiency} unit="%" highlight />
                </div>
            </div>

            {/* 温度 */}
            <div className="p-2 pt-0">
                <SectionTitle title="🌡️ 温度" color="from-orange-600 to-red-700" />
                <div className="bg-slate-900/40 border border-slate-700/30 border-t-0 rounded-b p-2 space-y-1.5">
                    <DataBox label="电堆温度" value={data.sensors.stackTemp} unit="℃" />
                    <DataBox label="环境温度" value={data.sensors.ambientTemp} unit="℃" />
                    <DataBox label="DCF温度" value={data.io.dcfMosTemp} unit="℃" />
                </div>
            </div>

            {/* 压力 */}
            <div className="p-2 pt-0">
                <SectionTitle title="💧 氢气压力" color="from-teal-600 to-cyan-700" />
                <div className="bg-slate-900/40 border border-slate-700/30 border-t-0 rounded-b p-2 space-y-1.5">
                    <DataBox label="氢气瓶" value={data.sensors.h2CylinderPressure} unit="MPa" />
                    <DataBox label="进口压力" value={data.sensors.h2InletPressure} unit="MPa" />
                    <DataBox label="氢气浓度" value={data.sensors.h2Concentration} unit="%vol" />
                </div>
            </div>

            {/* IO状态 */}
            <div className="p-2 pt-0">
                <SectionTitle title="🔌 IO状态" color="from-purple-600 to-indigo-700" />
                <div className="bg-slate-900/40 border border-slate-700/30 border-t-0 rounded-b p-2 space-y-1.5">
                    <StatusLED label="氢气进气阀" active={data.io.h2InletValve} activeColor="bg-cyan-400 shadow-cyan-400/50" />
                    <StatusLED label="排氢阀" active={data.io.h2PurgeValve} activeColor="bg-cyan-400 shadow-cyan-400/50" />
                    <StatusLED label="加热器" active={data.io.heater} activeColor="bg-red-500 shadow-red-500/50" />
                    <StatusLED label="风扇1" active={data.io.fan1} activeColor="bg-blue-500 shadow-blue-500/50" />
                    <StatusLED label="风扇2" active={data.io.fan2} activeColor="bg-blue-500 shadow-blue-500/50" />
                    <DataBox label="风扇1占空比" value={data.io.fan1Duty} unit="%" />
                </div>
            </div>

        </div>
    );
};
