
import React from 'react';
import { MachineState } from '../types';
import { Wind, Activity, Flame, Battery, Zap, Droplets } from 'lucide-react';


// Configuration for uniform styling
const STYLES = {
    font: {
        // Floating labels under icons (e.g. "氢气电磁阀")
        floatingLabel: "text-[11px] font-bold text-slate-700 whitespace-nowrap bg-transparent border-none px-1 rounded shadow-none",
        // Metric values inside components (e.g. DCF V/A)
        metricValue: "font-mono text-xs text-slate-300 leading-none font-bold",
        // Metric keys inside components (e.g. "输出电压")
        metricKey: "text-[11px] text-slate-200",
        // Big values inside Stack
        stackValue: "text-2xl font-mono leading-none tracking-tight font-bold",
        stackLabel: "text-[11px] text-slate-500 mb-0.5",
        // In-component labels (e.g. "H2" cylinder text)
        componentInnerLabel: "text-xs font-bold text-slate-300",
    },
    container: {
        // Dark component containers - increased z-index to sit above pipes
        darkBox: "bg-slate-700/90 border border-slate-500 shadow-[0_0_20px_rgba(0,0,0,0.15)] relative z-30",
    }
};

interface Props {
    data: MachineState;
}

export const SchematicView: React.FC<Props> = ({ data }) => {
    const h2Flowing = data.io.h2InletValve;
    const airFlowing = data.io.fan1;
    const fan2Running = data.io.fan2;
    const electricFlowing = data.power.stackCurrent > 1;

    // Colors
    // Colors
    const h2Color = h2Flowing ? 'bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.6)]' : 'bg-slate-700';
    const airColor = airFlowing ? 'bg-sky-500 shadow-[0_0_8px_rgba(14,165,233,0.6)]' : 'bg-slate-700';
    const fan2Color = fan2Running ? 'bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.6)]' : 'bg-slate-700';
    const elecColor = electricFlowing ? 'bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.6)]' : 'bg-slate-700';
    const loadColor = electricFlowing ? 'bg-emerald-600 shadow-[0_0_8px_rgba(5,150,105,0.6)]' : 'bg-slate-700';

    // Sub-components for continuous lines
    const Pipe = ({ className, color, vertical = false }: { className?: string, color: string, vertical?: boolean }) => (
        <div className={`${vertical ? 'w-1.5' : 'h-1.5'} ${color} transition-colors duration-500 rounded-none relative z-0 ${className}`} />
    );

    const Wire = ({ className, color }: { className?: string, color: string }) => (
        <div className={`h-2 ${color} transition-colors duration-300 shadow-sm relative z-0 ${className}`} />
    );

    return (
        <div className="relative w-full h-[380px] bg-slate-300 rounded-xl border border-slate-400 overflow-hidden shadow-sm flex items-center justify-center select-none">

            {/* Background Grid */}
            <div className="absolute inset-0 opacity-[0.20] pointer-events-none"
                style={{ backgroundImage: 'radial-gradient(#64748b 1px, transparent 1px)', backgroundSize: '24px 24px' }}>
            </div>

            {/* Main Flow Container - Flex Row with NO GAP to ensure connections */}
            <div className="flex items-center justify-center relative z-10 w-full max-w-6xl px-4">

                {/* === LEFT INPUTS === */}
                <div className="flex flex-col gap-10">

                    {/* TOP LINE: H2 -> Valve -> Stack */}
                    <div className="flex items-center gap-0">
                        {/* Cylinder */}
                        <div className="relative z-10 flex flex-col items-center group">
                            <div className={`w-12 h-16 rounded-lg flex items-center justify-center relative overflow-hidden z-10 ${STYLES.container.darkBox}`}>
                                <div className="absolute inset-x-0 bottom-0 bg-cyan-900/50 h-2/3 transition-all duration-1000" style={{ height: `${Math.min(100, data.sensors.h2CylinderPressure * 5)}%` }}></div>
                                <span className={`${STYLES.font.componentInnerLabel}`}>H2</span>
                            </div>
                            <div className={`absolute -bottom-5 w-full text-center font-mono font-bold bg-slate-300/50 border border-slate-400 px-1 rounded shadow-sm text-[10px] text-cyan-700`}>{data.sensors.h2CylinderPressure} Mpa</div>
                        </div>

                        <Pipe className="w-8 -mx-0.5 relative z-20" color={h2Color} />

                        {/* Solenoid Valve */}
                        <div className="relative z-10 flex flex-col items-center -mx-0.5">
                            <div className={`w-10 h-10 rounded-lg border flex items-center justify-center transition-all duration-300 z-10                                    ${data.io.h2InletValve
                                ? 'bg-slate-700/90 border-green-500 shadow-[0_0_15px_rgba(34,197,94,0.4)]'
                                : STYLES.container.darkBox}`}>
                                <div className={`w-1.5 h-6 rounded-full transition-all duration-300 
                                    ${data.io.h2InletValve ? 'rotate-90 bg-green-500 scale-110' : 'bg-slate-400'}`}></div>
                            </div>
                            <span className={`absolute -bottom-6 ${STYLES.font.floatingLabel}`}>氢气电磁阀</span>
                        </div>

                        <Pipe className="w-10" color={h2Color} />
                    </div>

                    {/* BOTTOM LINE: Fan -> Stack */}
                    <div className="flex items-center justify-end gap-0">
                        {/* Fan 1 */}
                        <div className="relative z-10 flex flex-col items-center">
                            <div className={`w-12 h-12 rounded-lg flex items-center justify-center relative z-10 ${data.io.fan1 ? 'border border-sky-500 bg-slate-700/90 shadow-lg shadow-sky-500/30' : STYLES.container.darkBox}`}>
                                <Wind className={`w-7 h-7 text-sky-500 ${data.io.fan1 ? 'animate-spin' : ''}`} />
                            </div>
                            <span className={`absolute -bottom-6 ${STYLES.font.floatingLabel}`}>风扇1 (供氧)</span>
                        </div>

                        <Pipe className="w-[5.5rem] relative -left-1 z-0" color={airColor} />
                    </div>
                </div>


                {/* === CENTER STACK === */}
                <div className="relative mx-[-2px] flex flex-col items-center">

                    {/* The Stack Box (Z-INDEX HIGHER than valve) */}
                    {/* The Stack Box (Z-INDEX HIGHER than valve) */}
                    {/* The Stack Box (Z-INDEX HIGHER than valve) */}
                    <div className={`w-60 h-48 rounded-xl flex flex-col relative z-30 ${STYLES.container.darkBox}`}>

                        {/* Stack Internals */}
                        <div className="flex-1 p-3 flex flex-col justify-between">
                            {/* Header */}
                            <div className="flex justify-between items-start">
                                <div>
                                    <span className="text-sm font-bold text-slate-200 flex items-center gap-2">
                                        <Activity className="w-4 h-4 text-cyan-400" /> 燃料电堆
                                    </span>                                </div>
                                {/* Heater Status */}
                                <div className="flex flex-col items-center p-1 rounded bg-slate-900 border border-slate-700">
                                    <Flame className={`w-3.5 h-3.5 ${data.io.heater ? 'text-red-500 animate-pulse' : 'text-slate-600'}`} />
                                    <span className="text-[8px] text-slate-200 mt-0.5">加热膜</span>
                                </div>
                            </div>

                            {/* Big Metrics */}
                            <div className="flex items-center justify-around bg-slate-900/50 rounded-lg p-2 border border-slate-700">
                                <div className="text-center">
                                    <div className={STYLES.font.stackLabel}>电堆温度</div>
                                    <div className={`${STYLES.font.stackValue} text-orange-500`}>{data.sensors.stackTemp} <span className="text-sm">°C</span></div>
                                </div>
                                <div className="w-px h-8 bg-slate-600"></div>
                                <div className="text-center">
                                    <div className={STYLES.font.stackLabel}>输出功率</div>
                                    <div className={`${STYLES.font.stackValue} text-amber-500`}>{data.power.stackPower} <span className="text-sm">kW</span></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Purge Valve (Positioned relative to stack bottom, but lower z-index) */}
                    <div className="absolute top-full flex flex-col items-center z-20 -mt-0.5">
                        {/* Pipe exiting stack */}
                        <div className="w-1.5 h-4 bg-slate-600"></div>
                        {/* Valve Icon */}
                        {/* Valve Icon */}
                        <div className={`w-8 h-6 rounded border flex items-center justify-center z-10 transition-all duration-300
                            ${data.io.h2PurgeValve ? 'bg-slate-700/90 border-green-500 shadow-[0_0_10px_rgba(34,197,94,0.3)]' : STYLES.container.darkBox}`}>
                            <Droplets className={`w-3.5 h-3.5 transition-colors duration-300 ${data.io.h2PurgeValve ? 'text-green-500 animate-bounce' : 'text-slate-500'}`} />
                        </div>
                        {/* Label */}
                        <span className={`mt-1.5 ${STYLES.font.floatingLabel}`}>排氢阀</span>
                    </div>
                </div>


                {/* === RIGHT OUTPUTS === */}
                <div className="flex items-center mx-[-2px]">

                    <Wire className="w-10" color={elecColor} />

                    {/* DCF Complex (Fan 2 + DCF) */}
                    <div className="relative flex flex-col items-center justify-center">

                        {/* DCF Unit - In Flow for Centering */}
                        <div className={`w-40 rounded-lg p-2 ${STYLES.container.darkBox}`}>
                            <div className="flex items-center justify-between mb-2 border-b border-slate-600 pb-1">
                                <span className="text-xs font-bold text-amber-500">DCF-DC</span>
                                <Zap className="w-3.5 h-3.5 text-amber-500" />
                            </div>
                            {/* DCF Metrics Grid */}
                            <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                                <div className="flex flex-col">
                                    <span className={STYLES.font.metricKey}>输出电压</span>
                                    <span className={STYLES.font.metricValue}>{data.power.dcfOutVoltage} V</span>
                                </div>
                                <div className="flex flex-col text-right">
                                    <span className={STYLES.font.metricKey}>输出电流</span>
                                    <span className={STYLES.font.metricValue}>{data.power.dcfOutCurrent} A</span>
                                </div>
                                <div className="flex flex-col">
                                    <span className={STYLES.font.metricKey}>温度</span>
                                    <span className={STYLES.font.metricValue}>{data.io.dcfMosTemp} °C</span>
                                </div>
                                <div className="flex flex-col text-right">
                                    <span className={STYLES.font.metricKey}>效率</span>
                                    <span className={`${STYLES.font.metricValue} text-emerald-500`}>{data.power.dcfEfficiency}%</span>
                                </div>
                            </div>
                        </div>

                        {/* Fan 2 (Cooling DCF) - Absolute Position Below */}
                        <div className="absolute top-full flex flex-col items-center mt-[-4px] z-20">
                            <Pipe className="h-4 w-1.5" color={fan2Color} vertical />
                            <div className={`w-10 h-10 rounded-lg flex items-center justify-center relative z-30 
                                ${data.io.fan2 ? 'border border-indigo-500 bg-slate-700/90 shadow-lg shadow-indigo-500/30' : STYLES.container.darkBox}`}>
                                <Wind className={`w-5 h-5 text-indigo-400 ${data.io.fan2 ? 'animate-spin' : ''}`} />
                            </div>
                            <div className="flex items-center gap-2 mt-1">
                                <span className={`${STYLES.font.floatingLabel} !shadow-none !bg-transparent`}>风扇2</span>
                            </div>
                        </div>
                    </div>

                    {/* Output Distribution (Parallel Split) */}
                    <div className="flex items-center -ml-1 gap-0">
                        {/* Main output wire from DCF */}
                        <Wire className="w-8 relative z-20" color={elecColor} />

                        {/* Splitter Node */}
                        <div className="flex flex-col justify-center gap-6 relative">
                            {/* Vertical Bus Bar connecting branches - Unified Color, Square Ends */}
                            <div className={`absolute left-0 top-[34px] bottom-[30px] w-2 ${elecColor} rounded-none`}></div>

                            {/* Top Branch: Battery */}
                            <div className="flex items-center pl-0">
                                <Wire className="w-8 -mr-1 relative z-20" color={elecColor} />
                                <div className="flex flex-col items-center ml-[-2px]">
                                    <div className={`w-16 h-16 border rounded flex items-center justify-center shadow-[0_0_20px_rgba(0,0,0,0.15)] transition-colors duration-500 relative z-30
                                        ${electricFlowing ? 'bg-slate-700/90 border-emerald-500/50 shadow-emerald-500/20' : STYLES.container.darkBox}`}>

                                        <Battery className={`w-6 h-8 ${electricFlowing ? 'text-emerald-500 fill-emerald-500/20 animate-pulse' : 'text-slate-400'}`} />
                                    </div>
                                    <span className={`absolute -top-4 ${STYLES.font.floatingLabel}`}>锂电池</span>
                                </div>
                            </div>

                            {/* Bottom Branch: DCL */}
                            <div className="flex items-center pl-0">
                                <Wire className="w-8 -mr-1 relative z-20" color={elecColor} />
                                <div className="flex flex-col items-center ml-[-2px]">
                                    <div className={`w-16 h-16 rounded-lg flex items-center justify-center relative z-20 ${STYLES.container.darkBox}`}>
                                        <span className="text-sm font-bold text-indigo-400">DCL</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
};
