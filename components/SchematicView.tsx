
import React from 'react';
import { MachineState } from '../types';
import { Wind, Activity, Flame, Battery, Zap, Droplets } from 'lucide-react';

interface Props {
    data: MachineState;
}

export const SchematicView: React.FC<Props> = ({ data }) => {
    const h2Flowing = data.io.h2InletValve;
    const airFlowing = data.io.fan1;
    const fan2Running = data.io.fan2;
    const electricFlowing = data.power.stackCurrent > 1;

    // Colors
    const h2Color = h2Flowing ? 'bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]' : 'bg-slate-700';
    const airColor = airFlowing ? 'bg-sky-400 shadow-[0_0_8px_rgba(56,189,248,0.6)]' : 'bg-slate-700';
    const fan2Color = fan2Running ? 'bg-indigo-400 shadow-[0_0_8px_rgba(129,140,248,0.6)]' : 'bg-slate-700';
    const elecColor = electricFlowing ? 'bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.6)]' : 'bg-slate-700';
    const loadColor = electricFlowing ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]' : 'bg-slate-700';

    // Sub-components for continuous lines
    const Pipe = ({ className, color, vertical = false }: { className?: string, color: string, vertical?: boolean }) => (
        <div className={`${vertical ? 'w-1.5' : 'h-1.5'} ${color} transition-colors duration-500 rounded-none relative z-0 ${className}`} />
    );

    const Wire = ({ className, color }: { className?: string, color: string }) => (
        <div className={`h-2 ${color} transition-colors duration-300 shadow-sm relative z-0 ${className}`} />
    );

    return (
        <div className="relative w-full h-[380px] bg-slate-900 rounded-xl border border-slate-700 overflow-hidden shadow-inner flex items-center justify-center select-none">

            {/* Background Grid */}
            <div className="absolute inset-0 opacity-10 pointer-events-none"
                style={{ backgroundImage: 'radial-gradient(#64748b 1px, transparent 1px)', backgroundSize: '24px 24px' }}>
            </div>

            {/* Main Flow Container - Flex Row with NO GAP to ensure connections */}
            <div className="flex items-center justify-center relative z-10 w-full max-w-6xl px-4">

                {/* === LEFT INPUTS === */}
                <div className="flex flex-col gap-10">

                    {/* TOP LINE: H2 -> Valve -> Stack */}
                    <div className="flex items-center">
                        {/* Cylinder */}
                        <div className="relative z-10 flex flex-col items-center group">
                            <div className="w-12 h-16 border-2 border-slate-600 bg-slate-800 rounded-lg flex items-center justify-center shadow-lg relative overflow-hidden z-10">
                                <div className="absolute inset-x-0 bottom-0 bg-cyan-900/40 h-2/3 transition-all duration-1000" style={{ height: `${Math.min(100, data.sensors.h2CylinderPressure * 5)}%` }}></div>
                                <span className="text-[10px] font-bold text-slate-400 -rotate-90">H2</span>
                            </div>
                            <div className="absolute -bottom-5 w-full text-center text-[9px] text-cyan-400 font-mono bg-slate-900/80 px-1 rounded">{data.sensors.h2CylinderPressure} Mpa</div>
                        </div>

                        <Pipe className="w-8" color={h2Color} />

                        {/* Solenoid Valve */}
                        <div className="relative z-10 flex flex-col items-center">
                            <div className={`w-10 h-10 rounded-lg border-2 flex items-center justify-center transition-all duration-300 z-10 
                                ${data.io.h2InletValve
                                    ? 'bg-green-500/20 border-green-400 shadow-[0_0_15px_rgba(74,222,128,0.4)]'
                                    : 'bg-slate-800 border-slate-600'}`}>
                                <div className={`w-1.5 h-5 rounded-full transition-all duration-300 
                                    ${data.io.h2InletValve ? 'rotate-90 bg-green-400 scale-110' : 'bg-slate-500'}`}></div>
                            </div>
                            <span className="absolute -bottom-5 text-[9px] text-slate-400 whitespace-nowrap bg-slate-900/80 px-1 rounded">氢气电磁阀</span>
                        </div>

                        <Pipe className="w-10" color={h2Color} />
                    </div>

                    {/* BOTTOM LINE: Fan -> Stack */}
                    <div className="flex items-center justify-end">
                        {/* Fan 1 */}
                        <div className="relative z-10 flex flex-col items-center">
                            <div className={`w-12 h-12 rounded-lg border-2 bg-slate-800 flex items-center justify-center relative z-10 ${data.io.fan1 ? 'border-sky-400 shadow-lg shadow-sky-500/20' : 'border-slate-600'}`}>
                                <Wind className={`w-7 h-7 text-sky-400 ${data.io.fan1 ? 'animate-spin' : ''}`} />
                            </div>
                            <span className="absolute -bottom-5 text-[9px] text-slate-400 whitespace-nowrap bg-slate-900/80 px-1 rounded">风扇1 (供氧)</span>
                        </div>

                        <Pipe className="w-[5.5rem]" color={airColor} />
                    </div>
                </div>


                {/* === CENTER STACK === */}
                <div className="relative mx-[-2px] flex flex-col items-center">

                    {/* The Stack Box (Z-INDEX HIGHER than valve) */}
                    <div className="w-56 h-44 bg-slate-800/95 backdrop-blur border-2 border-slate-600 rounded-xl shadow-2xl flex flex-col relative z-30">

                        {/* Stack Internals */}
                        <div className="flex-1 p-3 flex flex-col justify-between">
                            {/* Header */}
                            <div className="flex justify-between items-start">
                                <div>
                                    <span className="text-sm font-bold text-slate-200 flex items-center gap-2">
                                        <Activity className="w-4 h-4 text-cyan-400" /> 燃料电堆
                                    </span>
                                    <span className="text-[8px] text-slate-500">PEM Fuel Cell Stack</span>
                                </div>
                                {/* Heater Status */}
                                <div className="flex flex-col items-center p-1 rounded bg-slate-900/50 border border-slate-700/50">
                                    <Flame className={`w-3.5 h-3.5 ${data.io.heater ? 'text-red-500 animate-pulse' : 'text-slate-700'}`} />
                                    <span className="text-[8px] text-slate-500 mt-0.5">加热膜</span>
                                </div>
                            </div>

                            {/* Big Metrics */}
                            <div className="flex items-center justify-around bg-slate-900/50 rounded-lg p-2 border border-slate-700/50">
                                <div className="text-center">
                                    <div className="text-[9px] text-slate-500 mb-0.5">电堆温度</div>
                                    <div className="text-xl font-mono text-orange-400 leading-none tracking-tight">{data.sensors.stackTemp}</div>
                                </div>
                                <div className="w-px h-8 bg-slate-700"></div>
                                <div className="text-center">
                                    <div className="text-[9px] text-slate-500 mb-0.5">输出功率</div>
                                    <div className="text-xl font-mono text-amber-400 leading-none tracking-tight">{data.power.stackPower}</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Purge Valve (Positioned relative to stack bottom, but lower z-index) */}
                    <div className="absolute top-full flex flex-col items-center z-20 -mt-0.5">
                        {/* Pipe exiting stack */}
                        <div className="w-1.5 h-4 bg-slate-600"></div>
                        {/* Valve Icon */}
                        <div className={`w-8 h-6 rounded border flex items-center justify-center z-10 transition-all duration-300
                            ${data.io.h2PurgeValve ? 'bg-green-500/20 border-green-400 shadow-[0_0_10px_rgba(74,222,128,0.3)]' : 'bg-slate-800 border-slate-600'}`}>
                            <Droplets className={`w-3.5 h-3.5 transition-colors duration-300 ${data.io.h2PurgeValve ? 'text-green-400 animate-bounce' : 'text-slate-500'}`} />
                        </div>
                        {/* Label */}
                        <span className="text-[9px] text-slate-500 mt-1 whitespace-nowrap">排氢阀</span>
                    </div>
                </div>


                {/* === RIGHT OUTPUTS === */}
                <div className="flex items-center mx-[-2px]">

                    <Wire className="w-10" color={elecColor} />

                    {/* DCF (Displays All Metrics) */}
                    <div className="relative z-10">
                        <div className="w-36 bg-slate-800 border border-slate-600 rounded-lg p-2 shadow-lg z-10 relative">
                            <div className="flex items-center justify-between mb-2 border-b border-slate-700 pb-1">
                                <span className="text-[10px] font-bold text-amber-500">DCF</span>
                                <Zap className="w-3 h-3 text-amber-500" />
                            </div>
                            {/* DCF Metrics Grid */}
                            <div className="grid grid-cols-2 gap-x-3 gap-y-2">
                                <div className="flex flex-col">
                                    <span className="text-[8px] text-slate-500">输出电压</span>
                                    <span className="font-mono text-[10px] text-slate-300 leading-none">{data.power.dcfOutVoltage} V</span>
                                </div>
                                <div className="flex flex-col text-right">
                                    <span className="text-[8px] text-slate-500">输出电流</span>
                                    <span className="font-mono text-[10px] text-slate-300 leading-none">{data.power.dcfOutCurrent} A</span>
                                </div>
                                <div className="flex flex-col">
                                    <span className="text-[8px] text-slate-500">温度</span>
                                    <span className="font-mono text-[10px] text-slate-300 leading-none">{data.io.dcfMosTemp} °C</span>
                                </div>
                                <div className="flex flex-col text-right">
                                    <span className="text-[8px] text-slate-500">效率</span>
                                    <span className="font-mono text-[10px] text-emerald-400 leading-none">{data.power.dcfEfficiency}%</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <Wire className="w-8" color={elecColor} />

                    {/* DCL (Simple Block + Fan 2 below) */}
                    <div className="relative z-10 flex flex-col items-center">

                        {/* DCL Box */}
                        <div className="w-16 h-16 bg-slate-800 border border-slate-600 rounded-lg flex items-center justify-center shadow-lg relative z-20">
                            <span className="text-xs font-bold text-indigo-500">DCL</span>
                        </div>

                        {/* Connection to Fan 2 (Pipe going down) */}
                        <div className="absolute top-full flex flex-col items-center z-10">
                            <Pipe className="h-6" color={fan2Color} vertical />

                            {/* Fan 2 Component */}
                            <div className="flex flex-col items-center">
                                <div className={`w-10 h-10 rounded border bg-slate-800 flex items-center justify-center relative z-10 ${data.io.fan2 ? 'border-indigo-400 shadow-lg shadow-indigo-500/20' : 'border-slate-600'}`}>
                                    <Wind className={`w-5 h-5 text-indigo-400 ${data.io.fan2 ? 'animate-spin' : ''}`} />
                                </div>
                                <span className="text-[9px] text-slate-500 mt-1 whitespace-nowrap">风扇2</span>
                            </div>
                        </div>

                    </div>

                    <Wire className="w-10" color={loadColor} />

                    {/* Battery */}
                    <div className="relative z-10 flex flex-col items-center">
                        <div className={`w-14 h-20 bg-slate-800 border-2 rounded flex items-center justify-center shadow-lg transition-colors duration-500 ${electricFlowing ? 'border-emerald-500/50 shadow-emerald-500/20' : 'border-slate-600'}`}>
                            <Battery className={`w-6 h-8 ${electricFlowing ? 'text-emerald-400 fill-emerald-500/20 animate-pulse' : 'text-slate-600'}`} />
                        </div>
                        <span className="absolute -bottom-5 text-[9px] text-slate-400 font-bold bg-slate-900/80 px-1 rounded">锂电池</span>
                    </div>
                </div>

            </div>

        </div>
    );
};
