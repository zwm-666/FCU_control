
import React from 'react';
import { MachineState, SystemState } from '../types';

interface Props {
    data: MachineState;
}

export const IndustrialSchematic: React.FC<Props> = ({ data }) => {
    const h2Flowing = data.io.h2InletValve;
    const electricFlowing = data.power.stackCurrent > 1;

    // 工业扁平化配色
    const h2PipeColor = h2Flowing ? '#06B6D4' : '#94A3B8'; // cyan-500 / slate-400
    const airPipeColor = data.io.fan1 ? '#0EA5E9' : '#94A3B8'; // sky-500 / slate-400
    const elecWireColor = electricFlowing ? '#F59E0B' : '#94A3B8'; // amber-500 / slate-400

    return (
        <div className="flex-1 bg-gray-200 border border-slate-300 overflow-hidden relative rounded-lg m-2">
            {/* 标题栏 */}
            <div className="bg-slate-700 border-b border-slate-600 text-white text-sm font-bold px-4 py-2 flex justify-between items-center">
                <span className="flex items-center gap-2">
                    <span className="text-cyan-400">◈</span>
                    P&ID 工艺流程图
                </span>
                <span className={`px-3 py-1 rounded text-xs font-bold ${data.status.state === SystemState.RUN ? 'bg-cyan-500 text-white' :
                    data.status.state === SystemState.FAULT ? 'bg-red-500 text-white animate-pulse' :
                        data.status.state === SystemState.START ? 'bg-amber-500 text-white' :
                            'bg-slate-500 text-white'
                    }`}>
                    {data.status.state === SystemState.OFF ? '待机' :
                        data.status.state === SystemState.START ? '启动中' :
                            data.status.state === SystemState.RUN ? '运行中' :
                                data.status.state === SystemState.FAULT ? '故障' : '未知'}
                </span>
            </div>

            {/* SVG 工业扁平化流程图 */}
            <svg viewBox="0 0 900 400" className="w-full h-[calc(100%-44px)]" preserveAspectRatio="xMidYMid meet">
                {/* 浅灰色背景 */}
                <rect width="100%" height="100%" fill="#E5E7EB" />

                {/* === 左侧：氢气系统 === */}
                {/* 氢气瓶 */}
                <g transform="translate(30, 100)">
                    <rect x="0" y="0" width="70" height="100" rx="6"
                        fill="#ECFEFF" stroke="#334155" strokeWidth="2.5"
                    />
                    {/* 液位指示 */}
                    <rect x="5" y={95 - data.sensors.h2CylinderPressure * 4} width="60" height={data.sensors.h2CylinderPressure * 4}
                        fill="#06B6D4" opacity="0.3" />
                    <text x="35" y="35" textAnchor="middle" className="text-2xl fill-slate-700 font-bold">H₂</text>
                    <text x="35" y="55" textAnchor="middle" className="text-[10px] fill-slate-600">氢气瓶</text>
                    {/* 压力值 */}
                    <rect x="10" y="65" width="50" height="22" rx="3" fill="white" stroke="#06B6D4" strokeWidth="1.5" />
                    <text x="35" y="75" textAnchor="middle" className="text-[9px] fill-slate-500">储罐压力</text>
                    <text x="35" y="85" textAnchor="middle" className="text-xs fill-cyan-600 font-mono font-bold">
                        {data.sensors.h2CylinderPressure.toFixed(2)}
                    </text>
                </g>

                {/* 氢气管线1 - 从瓶子右侧到阀门 */}
                <line x1="100" y1="150" x2="130" y2="150" stroke={h2PipeColor} strokeWidth="5" strokeLinecap="round" />

                {/* 氢气进气阀 */}
                <g transform="translate(130, 134)">
                    <circle cx="16" cy="16" r="16" fill="white" stroke="#334155" strokeWidth="2.5" />
                    {/* X形阀门符号 */}
                    <line x1="8" y1="8" x2="24" y2="24" stroke={h2Flowing ? '#10B981' : '#94A3B8'} strokeWidth="3" strokeLinecap="round" />
                    <line x1="24" y1="8" x2="8" y2="24" stroke={h2Flowing ? '#10B981' : '#94A3B8'} strokeWidth="3" strokeLinecap="round" />
                    <text x="16" y="44" textAnchor="middle" className="text-[9px] fill-slate-700 font-bold">氢气阀</text>
                    <text x="16" y="-4" textAnchor="middle" className={`text-[9px] font-bold ${h2Flowing ? 'fill-green-600' : 'fill-slate-500'}`}>
                        {h2Flowing ? 'OPEN' : 'CLOSE'}
                    </text>
                </g>

                {/* 氢气管线2 - 从阀门到电堆 */}
                <line x1="162" y1="150" x2="220" y2="150" stroke={h2PipeColor} strokeWidth="5" strokeLinecap="round" />

                {/* 氢气入口压力测点 */}
                <g transform="translate(172, 108)">
                    <rect x="0" y="0" width="60" height="26" rx="4" fill="white" stroke="#0EA5E9" strokeWidth="1.5" />
                    <text x="30" y="11" textAnchor="middle" className="text-[8px] fill-slate-500">进口压力</text>
                    <text x="30" y="22" textAnchor="middle" className="text-[11px] fill-sky-600 font-mono font-bold">
                        {data.sensors.h2InletPressure.toFixed(2)} MPa
                    </text>
                </g>

                {/* === 下方：空气/氧气系统 === */}
                {/* 风扇1 (供氧) */}
                <g transform="translate(125, 230)">
                    <circle cx="30" cy="30" r="28" fill="white" stroke="#334155" strokeWidth="2.5" />
                    {/* 风扇叶片 */}
                    <g transform="translate(30, 30)">
                        {data.io.fan1 && (
                            <animateTransform
                                attributeName="transform"
                                type="rotate"
                                from="0 0 0"
                                to="360 0 0"
                                dur="0.8s"
                                repeatCount="indefinite"
                            />
                        )}
                        <path d="M0,-12 L-4,-8 L0,-6 L4,-8 Z" fill={data.io.fan1 ? '#0EA5E9' : '#CBD5E1'} />
                        <path d="M0,-12 L-4,-8 L0,-6 L4,-8 Z" fill={data.io.fan1 ? '#0EA5E9' : '#CBD5E1'} transform="rotate(120)" />
                        <path d="M0,-12 L-4,-8 L0,-6 L4,-8 Z" fill={data.io.fan1 ? '#0EA5E9' : '#CBD5E1'} transform="rotate(240)" />
                        <circle cx="0" cy="0" r="5" fill={data.io.fan1 ? '#0284C7' : '#94A3B8'} />
                    </g>
                    <text x="30" y="72" textAnchor="middle" className="text-[10px] fill-slate-700 font-bold">风扇1 (供氧)</text>
                </g>

                {/* 空气管线 - 从风扇到电堆 */}
                <line x1="183" y1="260" x2="220" y2="260" stroke={airPipeColor} strokeWidth="5" strokeLinecap="round" />

                {/* === 中央：燃料电堆 === */}
                <g transform="translate(220, 60)">
                    <rect x="0" y="0" width="220" height="260" rx="8"
                        fill="white" stroke="#334155" strokeWidth="3"
                    />

                    {/* 加热膜指示 */}
                    <rect x="165" y="10" width="48" height="24" rx="4" fill={data.io.heater ? '#EF4444' : '#CBD5E1'} />
                    <text x="189" y="26" textAnchor="middle" className="text-[10px] fill-white font-bold">加热膜</text>

                    {/* 标题 */}
                    <text x="110" y="30" textAnchor="middle" className="text-base fill-slate-700 font-bold">供试品 (燃料电堆)</text>

                    {/* 内部数据板 */}
                    <rect x="20" y="50" width="180" height="140" rx="6" fill="#F8FAFC" stroke="#CBD5E1" strokeWidth="2" />

                    {/* 电堆温度 */}
                    <text x="110" y="75" textAnchor="middle" className="text-xs fill-slate-600">电堆温度</text>
                    <text x="110" y="100" textAnchor="middle" className="text-3xl fill-orange-500 font-mono font-bold">
                        {data.sensors.stackTemp.toFixed(1)}
                    </text>
                    <text x="110" y="115" textAnchor="middle" className="text-sm fill-orange-500">℃</text>

                    <line x1="30" y1="130" x2="190" y2="130" stroke="#CBD5E1" strokeWidth="1.5" />

                    {/* 输出功率 */}
                    <text x="110" y="150" textAnchor="middle" className="text-xs fill-slate-600">输出功率</text>
                    <text x="110" y="175" textAnchor="middle" className="text-3xl fill-cyan-600 font-mono font-bold">
                        {data.power.stackPower.toFixed(2)}
                    </text>
                    <text x="110" y="188" textAnchor="middle" className="text-sm fill-cyan-600">kW</text>

                    {/* 电堆电压电流 */}
                    <g transform="translate(20, 205)">
                        <rect x="0" y="0" width="85" height="38" rx="4" fill="white" stroke="#94A3B8" strokeWidth="1" />
                        <text x="42" y="14" textAnchor="middle" className="text-[9px] fill-slate-500">电堆电压</text>
                        <text x="42" y="30" textAnchor="middle" className="text-sm fill-slate-700 font-mono font-bold">
                            {data.power.stackVoltage.toFixed(1)} V
                        </text>
                    </g>
                    <g transform="translate(115, 205)">
                        <rect x="0" y="0" width="85" height="38" rx="4" fill="white" stroke="#94A3B8" strokeWidth="1" />
                        <text x="42" y="14" textAnchor="middle" className="text-[9px] fill-slate-500">电堆电流</text>
                        <text x="42" y="30" textAnchor="middle" className="text-sm fill-slate-700 font-mono font-bold">
                            {data.power.stackCurrent.toFixed(1)} A
                        </text>
                    </g>
                </g>

                {/* 排氢阀 - 连接到电堆底部 */}
                <g transform="translate(320, 320)">
                    <line x1="10" y1="-10" x2="10" y2="10" stroke={data.io.h2PurgeValve ? '#06B6D4' : '#CBD5E1'} strokeWidth="4" />
                    <circle cx="10" cy="20" r="12" fill="white" stroke="#334155" strokeWidth="2" />
                    <path d="M10,14 L6,20 L10,26 L14,20 Z" fill={data.io.h2PurgeValve ? '#10B981' : '#94A3B8'} />
                    <text x="10" y="46" textAnchor="middle" className="text-[9px] fill-slate-700 font-bold">排氢阀</text>
                </g>

                {/* === 右侧：电力输出系统 === */}
                {/* 电力线1 - 从电堆到DCF */}
                <line x1="440" y1="190" x2="480" y2="190" stroke={elecWireColor} strokeWidth="5" strokeLinecap="round" />

                {/* DCF-DC变换器 */}
                <g transform="translate(480, 90)">
                    <rect x="0" y="0" width="140" height="200" rx="6"
                        fill="white" stroke="#334155" strokeWidth="3"
                    />
                    <text x="70" y="25" textAnchor="middle" className="text-base fill-amber-600 font-bold">DCF-DC</text>

                    <rect x="15" y="40" width="110" height="145" rx="4" fill="#FFFBEB" stroke="#FCD34D" strokeWidth="1.5" />

                    {/* DCF参数 */}
                    <g transform="translate(15, 50)">
                        <text x="55" y="15" textAnchor="middle" className="text-[10px] fill-slate-600">输出电压</text>
                        <text x="55" y="32" textAnchor="middle" className="text-lg fill-amber-600 font-mono font-bold">
                            {data.power.dcfOutVoltage.toFixed(1)} V
                        </text>

                        <text x="55" y="55" textAnchor="middle" className="text-[10px] fill-slate-600">输出电流</text>
                        <text x="55" y="72" textAnchor="middle" className="text-lg fill-blue-600 font-mono font-bold">
                            {data.power.dcfOutCurrent.toFixed(1)} A
                        </text>

                        <text x="55" y="95" textAnchor="middle" className="text-[10px] fill-slate-600">MOS温度</text>
                        <text x="55" y="112" textAnchor="middle" className="text-lg fill-orange-500 font-mono font-bold">
                            {data.io.dcfMosTemp.toFixed(0)} ℃
                        </text>

                        <text x="55" y="133" textAnchor="middle" className="text-[10px] fill-slate-600">转换效率</text>
                        <text x="55" y="148" textAnchor="middle" className="text-base fill-green-600 font-mono font-bold">
                            {data.power.dcfEfficiency.toFixed(0)}%
                        </text>
                    </g>
                </g>

                {/* 风扇2 (DCF散热) */}
                <g transform="translate(520, 300)">
                    <circle cx="25" cy="25" r="24" fill="white" stroke="#334155" strokeWidth="2.5" />
                    <g transform="translate(25, 25)">
                        {data.io.fan2 && (
                            <animateTransform
                                attributeName="transform"
                                type="rotate"
                                from="0 0 0"
                                to="360 0 0"
                                dur="0.6s"
                                repeatCount="indefinite"
                            />
                        )}
                        <path d="M0,-10 L-3,-7 L0,-5 L3,-7 Z" fill={data.io.fan2 ? '#6366F1' : '#CBD5E1'} />
                        <path d="M0,-10 L-3,-7 L0,-5 L3,-7 Z" fill={data.io.fan2 ? '#6366F1' : '#CBD5E1'} transform="rotate(120)" />
                        <path d="M0,-10 L-3,-7 L0,-5 L3,-7 Z" fill={data.io.fan2 ? '#6366F1' : '#CBD5E1'} transform="rotate(240)" />
                        <circle cx="0" cy="0" r="4" fill={data.io.fan2 ? '#4F46E5' : '#94A3B8'} />
                    </g>
                    <text x="25" y="62" textAnchor="middle" className="text-[10px] fill-slate-700 font-bold">风扇2 (散热)</text>
                </g>

                {/* 电力线2 - 从DCF到分支点 */}
                <line x1="620" y1="190" x2="660" y2="190" stroke={elecWireColor} strokeWidth="5" strokeLinecap="round" />

                {/* 分支点 */}
                <circle cx="660" cy="190" r="5" fill={elecWireColor} />
                <line x1="660" y1="190" x2="660" y2="130" stroke={elecWireColor} strokeWidth="5" />
                <line x1="660" y1="190" x2="660" y2="250" stroke={elecWireColor} strokeWidth="5" />

                {/* 支路线 - 缩短距离 */}
                <line x1="660" y1="130" x2="690" y2="130" stroke={elecWireColor} strokeWidth="5" strokeLinecap="round" />
                <line x1="660" y1="250" x2="690" y2="250" stroke={elecWireColor} strokeWidth="5" strokeLinecap="round" />

                {/* 锂电池 */}
                <g transform="translate(690, 90)">
                    <rect x="0" y="0" width="90" height="80" rx="6"
                        fill="white" stroke="#10B981" strokeWidth="2.5"
                    />
                    <rect x="35" y="-8" width="20" height="10" rx="2" fill="#10B981" />
                    <text x="45" y="30" textAnchor="middle" className="text-xs fill-slate-600 font-bold">锂电池</text>
                    <text x="45" y="55" textAnchor="middle" className="text-3xl">🔋</text>
                </g>

                {/* 电子负载 (DCL) */}
                <g transform="translate(690, 210)">
                    <rect x="0" y="0" width="90" height="80" rx="6"
                        fill="white" stroke="#8B5CF6" strokeWidth="2.5"
                    />
                    <text x="45" y="40" textAnchor="middle" className="text-xl fill-purple-600 font-bold">DCL</text>
                    <text x="45" y="60" textAnchor="middle" className="text-[10px] fill-slate-600">电子负载</text>
                </g>

            </svg>
        </div>
    );
};
