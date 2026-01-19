
import React from 'react';
import { MachineState, SystemState } from '../types';
import { AdvancedPipe } from './AdvancedPipe';
import { IndustrialValve } from './IndustrialValve';

interface Props {
    data: MachineState;
}

export const IndustrialSchematic: React.FC<Props> = ({ data }) => {
    const h2Flowing = data.io.h2InletValve;
    const electricFlowing = data.power.stackCurrent > 1;
    const isRunning = data.status.state === SystemState.RUN;

    // 发光颜色定义
    const glowCyan = '#00F5FF';
    const glowGreen = '#00FF88';
    const glowAmber = '#FFB800';
    const dimColor = '#3a5070';

    // 动态颜色 - 管道始终可见，开启时发光
    const h2PipeColor = h2Flowing ? glowCyan : dimColor;
    const elecColor = electricFlowing ? glowAmber : dimColor;
    // 风扇1视觉逻辑：开关开启 或 转速>0
    const isFan1On = data.io.fan1 || (data.io.fan1Duty && data.io.fan1Duty > 0);
    const airPipeColor = isFan1On ? glowCyan : dimColor;
    // 储氢罐视觉逻辑：有压力即发光
    const hasH2Pressure = data.sensors.h2CylinderPressure > 0.5;

    // 字体大小定义 (统一管理)
    const FONT_SIZES = {
        labelSmall: 12,  // 原 7-9
        labelMedium: 14, // 原 9-11
        valueLarge: 14,  // 原 11-15 (数据值)
        valueHuge: 32,   // 原 22-24 (大标题/大数值)
        icon: 28         // 原 24 (图标)
    };

    return (
        <div className="flex-1 overflow-hidden relative rounded-lg m-2"
            style={{
                background: 'linear-gradient(135deg, #050b14 0%, #0a1525 100%)',
                boxShadow: '0 20px 50px rgba(0,0,0,0.5), inset 0 0 0 1px rgba(255,255,255,0.05)'
            }}>

            {/* 玻璃拟态外部光晕 */}
            <div className="absolute inset-0 rounded-lg pointer-events-none"
                style={{ background: 'radial-gradient(circle at 50% -20%, rgba(0,245,255,0.15), transparent 70%)' }} />

            <svg viewBox="0 0 900 400" className="w-full h-full" preserveAspectRatio="xMidYMid meet">
                {/* 滤镜定义 - 强化发光效果 */}
                <defs>
                    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur" />
                        <feColorMatrix in="blur" type="matrix" values="
                            1 0 0 0 0
                            0 1 0 0 0
                            0 0 1 0 0
                            0 0 0 1.5 0" result="brightBlur" />
                        <feMerge>
                            <feMergeNode in="brightBlur" />
                            <feMergeNode in="brightBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                    <filter id="pipeBlur" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur in="SourceGraphic" stdDeviation="8" />
                    </filter>
                    {/* 规范一：heavy-blur 滤镜用于管道环境光晕 */}
                    <filter id="heavy-blur" x="-100%" y="-100%" width="300%" height="300%">
                        <feGaussianBlur in="SourceGraphic" stdDeviation="10" />
                    </filter>

                    {/* 玻璃拟态专用渐变 */}
                    <linearGradient id="glassMainGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor="#1a3a5c" stopOpacity="0.4" />
                        <stop offset="50%" stopColor="#0d2137" stopOpacity="0.6" />
                        <stop offset="100%" stopColor="#051525" stopOpacity="0.8" />
                    </linearGradient>

                    <linearGradient id="glassBorderGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="rgba(255,255,255,0.6)" />
                        <stop offset="20%" stopColor="rgba(0,245,255,0.3)" />
                        <stop offset="100%" stopColor="rgba(255,255,255,0.05)" />
                    </linearGradient>

                    <linearGradient id="glassHighlightGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="rgba(255,255,255,0.4)" />
                        <stop offset="40%" stopColor="rgba(255,255,255,0.1)" />
                        <stop offset="100%" stopColor="transparent" />
                    </linearGradient>
                </defs>

                {/* 全局玻璃罩 - 模拟参考图的厚实玻璃感 */}
                {/* 1. 主体底色 */}
                <rect x="25" y="25" width="850" height="350" rx="24"
                    fill="url(#glassMainGradient)" stroke="none" />

                {/* 2. 顶部高光反射 (Gloss Reflection) */}
                <path d="M 25 49 A 24 24 0 0 1 49 25 L 851 25 A 24 24 0 0 1 875 49 L 875 140 C 600 200, 300 50, 25 180 Z"
                    fill="url(#glassHighlightGradient)" opacity="0.5" style={{ mixBlendMode: 'overlay' }} />

                {/* 3. 玻璃边缘 (Rim Light) */}
                <rect x="25" y="25" width="850" height="350" rx="24"
                    fill="none" stroke="url(#glassBorderGradient)" strokeWidth="2" />

                {/* 4. 底部反光强调 (Rim Light Bottom) */}
                <path d="M 50 375 L 850 375" stroke="rgba(0,245,255,0.3)" strokeWidth="2" strokeLinecap="round" filter="url(#glow)" />

                {/* ========== 左侧：氢气系统 ========== */}

                {/* 储氢罐 - 规范二：玻璃质感 */}
                <g transform="translate(30, 100)">
                    {/* 主体背景 - 加宽以容纳数值 */}
                    <rect x="0" y="0" width="90" height="100" rx="8"
                        fill="rgba(0, 0, 0, 0.5)"
                        stroke={hasH2Pressure ? glowCyan : 'rgba(255, 255, 255, 0.1)'}
                        strokeWidth={hasH2Pressure ? 2 : 1}
                        filter={hasH2Pressure ? 'url(#glow)' : 'none'} />
                    {/* 液位指示 */}
                    <rect x="5" y={95 - Math.min(data.sensors.h2CylinderPressure * 4, 85)}
                        width="80" height={Math.min(data.sensors.h2CylinderPressure * 4, 85)} rx="4"
                        fill={glowCyan} opacity="0.3" />
                    {/* 标签 */}
                    <text x="45" y="38" textAnchor="middle" fill={glowCyan} fontSize={FONT_SIZES.valueHuge} fontWeight="bold">H₂</text>
                    <text x="45" y="55" textAnchor="middle" fill="#94a3b8" fontSize={FONT_SIZES.labelSmall}>储氢罐</text>
                    {/* 数值 */}
                    <text x="45" y="85" textAnchor="middle" fill="#ffffff" fontSize={FONT_SIZES.valueLarge} fontWeight="bold"
                        fontFamily="'JetBrains Mono', 'Roboto Mono', monospace"
                        style={{ textShadow: 'none' }}>
                        {data.sensors.h2CylinderPressure.toFixed(2)} MPa
                    </text>
                </g>

                {/* 氢气管线1：储氢罐 → 进气阀 (调整起点适配加宽的罐体) */}
                <AdvancedPipe x1={120} y1={150} x2={150} y2={150} isActive={h2Flowing} />

                {/* 进气阀 (使用统一组件) */}
                <IndustrialValve
                    x={168} y={150}
                    isOpen={h2Flowing}
                    orientation="horizontal"
                    label="进氢阀"
                />

                {/* 氢气管线2：进气阀 → 电堆 (修正连接坐标) */}
                {/* 进气阀出口 150+36=186. 电堆(240)+10=250. */}
                <AdvancedPipe x1={186} y1={150} x2={250} y2={150} isActive={h2Flowing} />

                {/* 进口压力传感器 (带玻璃罩 - 右移) */}
                <g transform="translate(210, 100)">
                    <rect x="-5" y="-5" width="40" height="40" rx="4" fill="rgba(0,0,0,0.3)" stroke="none" />
                    <circle cx="15" cy="15" r="12" fill="none" stroke={h2Flowing ? glowCyan : dimColor} strokeWidth="2" />
                    <line x1="15" y1="15" x2="22" y2="8" stroke={h2Flowing ? glowCyan : dimColor} strokeWidth="2" />
                    <text x="15" y="-12" textAnchor="middle" fill="#64B5F6" fontSize={FONT_SIZES.labelSmall}>进口压力</text>
                    <text x="50" y="18" textAnchor="start" fill="#ffffff" fontSize={FONT_SIZES.valueLarge - 4} fontFamily="monospace">
                        {data.sensors.h2InletPressure.toFixed(2)} MPa
                    </text>
                </g>

                {/* ========== 下方：供氧风扇 ========== */}
                {/* 右移以匹配上部布局 */}
                <g transform="translate(140, 230)">
                    <circle cx="25" cy="25" r="24" fill="url(#glassGradient)" stroke={isFan1On ? glowCyan : '#3a5070'} strokeWidth="2"
                        filter={isFan1On ? 'url(#glow)' : 'none'} />
                    {/* 风扇叶片 - 使用CSS动画 */}
                    <g className={isFan1On ? 'animate-spin' : ''} style={{ transformOrigin: '25px 25px', animationDuration: '0.8s' }}>
                        <polygon points="25,13 19,29 31,29" fill={isFan1On ? glowCyan : '#3a5070'} />
                        <polygon points="25,13 19,29 31,29" fill={isFan1On ? glowCyan : '#3a5070'} style={{ transform: 'rotate(120deg)', transformOrigin: '25px 25px' }} />
                        <polygon points="25,13 19,29 31,29" fill={isFan1On ? glowCyan : '#3a5070'} style={{ transform: 'rotate(240deg)', transformOrigin: '25px 25px' }} />
                    </g>
                    <text x="25" y="68" textAnchor="middle" fill="#64B5F6" fontSize={FONT_SIZES.labelSmall}>风扇1(空气)</text>
                </g>

                {/* 空气管线：风扇 → 电堆 */}
                <AdvancedPipe x1={189} y1={255} x2={240} y2={255} isActive={!!isFan1On} />

                {/* ========== 中央：燃料电堆 (心电图风格 + 修复框大小与对齐) ========== */}
                {/* 整体右移 20px (220->240) 以避让左侧挤压 */}
                <g transform="translate(240, 60)">
                    {/* 标题 */}
                    <text x="105" y="20" textAnchor="middle" fill="#64B5F6" fontSize={FONT_SIZES.labelMedium} fontWeight="bold">H2-FC 电堆</text>

                    {/* 玻璃外框 - 扩大以包含底部文字 */}
                    {/* 原 height=120, 现增加到 190 以覆盖底部 y=160+ 的文字 */}
                    <rect x="0" y="30" width="210" height="190" rx="8"
                        fill="rgba(0, 0, 0, 0.5)"
                        stroke={isRunning ? glowCyan : 'rgba(255, 255, 255, 0.15)'}
                        strokeWidth={isRunning ? 2 : 1}
                        filter={isRunning ? 'url(#glow)' : 'none'} />

                    {/* 左侧触点区域 (2个触点) */}
                    {/* 调整对齐：进气管在 y=150. Stack y=60. Relative Pipe Y=90. */}
                    {/* 触点组在 y=45. 需要其中一个触点在 relative y=45 (45+45=90). */}
                    <g transform="translate(10, 45)">
                        {/* 上触点 */}
                        <rect x="0" y="0" width="20" height="35" rx="3" fill="#1e293b" stroke="#475569" strokeWidth="1" />
                        <circle cx="10" cy="17" r="6" fill={isRunning ? glowCyan : '#334155'} filter={isRunning ? 'url(#glow)' : 'none'} />
                        {/* 下触点 - 调整位置对齐管道 (Center at 45) */}
                        <rect x="0" y="28" width="20" height="35" rx="3" fill="#1e293b" stroke="#475569" strokeWidth="1" />
                        <circle cx="10" cy="45" r="6" fill={isRunning ? glowCyan : '#334155'} filter={isRunning ? 'url(#glow)' : 'none'} />
                        <text x="10" y="75" textAnchor="middle" fill="#64B5F6" fontSize={FONT_SIZES.labelSmall - 2}>H₂ IN</text>
                    </g>

                    {/* 中央心电图区域 */}
                    <g transform="translate(40, 45)">
                        {/* 背景 */}
                        <rect x="0" y="0" width="130" height="90" rx="4" fill="rgba(0, 20, 40, 0.7)" stroke="rgba(0, 229, 255, 0.2)" strokeWidth="1" />

                        {/* 网格线 */}
                        {Array.from({ length: 6 }).map((_, i) => (
                            <line key={`H${i}`} x1="0" y1={15 * i} x2="130" y2={15 * i} stroke="rgba(0, 229, 255, 0.1)" strokeWidth="1" />
                        ))}
                        {Array.from({ length: 9 }).map((_, i) => (
                            <line key={`V${i}`} x1={15 * i} y1="0" x2={15 * i} y2="90" stroke="rgba(0, 229, 255, 0.1)" strokeWidth="1" />
                        ))}

                        {/* 心电图/闪电波形 */}
                        {isRunning ? (
                            <g>
                                <path d="M 0 45 L 20 45 L 25 45 L 30 30 L 35 60 L 40 20 L 45 70 L 50 45 L 70 45 L 75 45 L 80 30 L 85 60 L 90 20 L 95 70 L 100 45 L 130 45"
                                    stroke={glowCyan} strokeWidth="3" fill="none" opacity="0.4" filter="url(#heavy-blur)" />
                                <path d="M 0 45 L 20 45 L 25 45 L 30 30 L 35 60 L 40 20 L 45 70 L 50 45 L 70 45 L 75 45 L 80 30 L 85 60 L 90 20 L 95 70 L 100 45 L 130 45"
                                    stroke={glowCyan} strokeWidth="2" fill="none" filter="url(#glow)">
                                    <animate attributeName="stroke-dashoffset" from="0" to="-260" dur="1.5s" repeatCount="indefinite" />
                                </path>
                                <path d="M 0 45 L 20 45 L 25 45 L 30 30 L 35 60 L 40 20 L 45 70 L 50 45 L 70 45 L 75 45 L 80 30 L 85 60 L 90 20 L 95 70 L 100 45 L 130 45"
                                    stroke="#ffffff" strokeWidth="1" fill="none" opacity="0.6"
                                    strokeDasharray="10 20">
                                    <animate attributeName="stroke-dashoffset" from="0" to="-30" dur="0.3s" repeatCount="indefinite" />
                                </path>
                            </g>
                        ) : (
                            <line x1="0" y1="45" x2="130" y2="45" stroke="#334155" strokeWidth="2" />
                        )}

                        {/* 功率显示 */}
                        <text x="65" y="80" textAnchor="middle" fill="#ffffff" fontSize={FONT_SIZES.labelMedium} fontWeight="bold"
                            fontFamily="'JetBrains Mono', monospace"
                            style={{ textShadow: isRunning ? '0 0 4px rgba(0, 229, 255, 0.8)' : 'none' }}>
                            {data.power.stackPower.toFixed(2)} kW
                        </text>
                    </g>

                    {/* 右侧触点区域 (2个触点) */}
                    <g transform="translate(180, 45)">
                        {/* 上触点 */}
                        <rect x="0" y="0" width="20" height="35" rx="3" fill="#1e293b" stroke="#475569" strokeWidth="1" />
                        <circle cx="10" cy="17" r="6" fill={electricFlowing ? glowCyan : '#334155'} filter={electricFlowing ? 'url(#glow)' : 'none'} />
                        {/* 下触点 - 对应左侧调整 */}
                        <rect x="0" y="28" width="20" height="35" rx="3" fill="#1e293b" stroke="#475569" strokeWidth="1" />
                        <circle cx="10" cy="45" r="6" fill={electricFlowing ? glowCyan : '#334155'} filter={electricFlowing ? 'url(#glow)' : 'none'} />
                        <text x="10" y="75" textAnchor="middle" fill="#64B5F6" fontSize={FONT_SIZES.labelSmall - 2}>DC OUT</text>
                    </g>


                    {/* 底部数据 */}
                    <g transform="translate(0, 160)">
                        <text x="45" y="5" textAnchor="middle" fill="#94a3b8" fontSize={FONT_SIZES.labelSmall}>电压</text>
                        <text x="45" y="26" textAnchor="middle" fill="#ffffff" fontSize={FONT_SIZES.valueLarge} fontWeight="bold" fontFamily="monospace">
                            {data.power.stackVoltage.toFixed(1)}V
                        </text>
                        <text x="105" y="5" textAnchor="middle" fill="#94a3b8" fontSize={FONT_SIZES.labelSmall}>电流</text>
                        <text x="105" y="26" textAnchor="middle" fill="#ffffff" fontSize={FONT_SIZES.valueLarge} fontWeight="bold" fontFamily="monospace">
                            {data.power.stackCurrent.toFixed(1)}A
                        </text>
                        <text x="165" y="5" textAnchor="middle" fill="#94a3b8" fontSize={FONT_SIZES.labelSmall}>温度</text>
                        <text x="165" y="26" textAnchor="middle" fill={data.sensors.stackTemp > 70 ? '#EF4444' : '#ffffff'} fontSize={FONT_SIZES.valueLarge} fontWeight="bold" fontFamily="monospace">
                            {data.sensors.stackTemp.toFixed(0)}°C
                        </text>
                    </g>
                </g>

                {/* 加热膜指示 (调整位置 - 增强可见性) */}
                <g transform="translate(380, 50)">
                    {/* 背景玻璃光晕 - 增加一个底座让其更明显 */}
                    <rect x="-2" y="-2" width="48" height="24" rx="6" fill="rgba(0,0,0,0.5)" stroke="none" />

                    {/* 主体 */}
                    <rect x="0" y="0" width="44" height="20" rx="4"
                        fill={data.io.heater ? '#EF4444' : 'rgba(30, 41, 59, 0.9)'}
                        stroke={data.io.heater ? '#EF4444' : 'rgba(255, 255, 255, 0.3)'}
                        strokeWidth={data.io.heater ? 0 : 1}
                        filter={data.io.heater ? 'url(#glow)' : 'none'} />

                    <text x="22" y="14" textAnchor="middle" fill="white" fontSize={FONT_SIZES.labelSmall} fontWeight="bold">加热膜</text>
                </g>

                {/* 排氢阀 (使用统一组件) */}
                <IndustrialValve
                    x={325} y={310}
                    isOpen={data.io.h2PurgeValve}
                    orientation="vertical"
                    label="排氢阀"
                />
                {/* 替换为 AdvancedPipe 连接线 (305+20=325, 320-50=270 -> 320+15=335) */}
                <AdvancedPipe x1={325} y1={280} x2={325} y2={335} isActive={data.io.h2PurgeValve} />

                {/* ========== 右侧：电力系统 ========== */}

                {/* 电力线1：电堆 → DCF (修正连接) */}
                {/* Stack Right Contact Center: X=240+180+10=430. Pipe Start 440. */}
                {/* DCF starts X=480. Target relative to DCF Top(90) is 60. Abs Y=150. */}
                <AdvancedPipe x1={440} y1={150} x2={480} y2={150} isActive={electricFlowing} />

                {/* DCF-DC 变换器 - 规范二：玻璃质感 */}
                {/* 下移 25px (65 -> 90) 以与电堆中心对齐 */}
                <g transform="translate(480, 90)">
                    {/* 主容器：半透明黑 + 极细微亮边框 */}
                    <rect x="0" y="0" width="130" height="190" rx="8"
                        fill="rgba(0, 0, 0, 0.5)"
                        stroke={electricFlowing ? glowAmber : 'rgba(255, 255, 255, 0.1)'}
                        strokeWidth={electricFlowing ? 2 : 1}
                        filter={electricFlowing ? 'url(#glow)' : 'none'} />
                    <text x="65" y="22" textAnchor="middle" fill="#FFB800" fontSize={FONT_SIZES.labelMedium} fontWeight="bold">DCF-DC 变换器</text>

                    {/* 内部数据区 */}
                    <rect x="10" y="32" width="110" height="148" rx="5" fill="rgba(0, 0, 0, 0.4)" stroke="rgba(255, 255, 255, 0.1)" strokeWidth="1" />

                    {/* 规范三：科技字体 + 极小阴影 */}
                    <text x="65" y="52" textAnchor="middle" fill="#94a3b8" fontSize={FONT_SIZES.labelSmall}>输出电压</text>
                    <text x="65" y="70" textAnchor="middle" fill="#ffffff" fontSize={FONT_SIZES.valueLarge} fontWeight="bold"
                        fontFamily="'JetBrains Mono', 'Roboto Mono', monospace"
                        style={{ textShadow: electricFlowing ? '0 0 2px rgba(255, 184, 0, 0.8)' : 'none' }}>
                        {data.power.dcfOutVoltage.toFixed(1)} V
                    </text>

                    <text x="65" y="92" textAnchor="middle" fill="#94a3b8" fontSize={FONT_SIZES.labelSmall}>输出电流</text>
                    <text x="65" y="110" textAnchor="middle" fill="#ffffff" fontSize={FONT_SIZES.valueLarge} fontWeight="bold"
                        fontFamily="'JetBrains Mono', 'Roboto Mono', monospace"
                        style={{ textShadow: electricFlowing ? '0 0 2px rgba(0, 229, 255, 0.8)' : 'none' }}>
                        {data.power.dcfOutCurrent.toFixed(1)} A
                    </text>

                    <text x="65" y="132" textAnchor="middle" fill="#94a3b8" fontSize={FONT_SIZES.labelSmall}>DCF温度</text>
                    <text x="65" y="150" textAnchor="middle" fill={data.io.dcfMosTemp > 60 ? '#FF6B6B' : '#ffffff'} fontSize={FONT_SIZES.valueLarge} fontWeight="bold"
                        fontFamily="'JetBrains Mono', 'Roboto Mono', monospace">
                        {data.io.dcfMosTemp.toFixed(0)} ℃
                    </text>

                    <text x="65" y="172" textAnchor="middle" fill="#ffffff" fontSize={FONT_SIZES.labelMedium} fontWeight="bold"
                        fontFamily="'JetBrains Mono', 'Roboto Mono', monospace"
                        style={{ textShadow: '0 0 2px rgba(0, 255, 136, 0.8)' }}>
                        效率 {data.power.dcfEfficiency.toFixed(0)}%
                    </text>
                </g>



                {/* 电力线2：DCF → 分支点 (玻璃管道) */}
                {/* DCF Center Y moved 160 -> 185 */}
                <AdvancedPipe x1={610} y1={185} x2={650} y2={185} isActive={electricFlowing} />

                {/* 垂直分流线 (全长) */}
                {/* 100->125, 220->245 */}
                <AdvancedPipe x1={650} y1={125} x2={650} y2={245} isActive={electricFlowing} />

                {/* 水平分流线 (Top) */}
                <AdvancedPipe x1={650} y1={125} x2={690} y2={125} isActive={electricFlowing} />

                {/* 水平分流线 (Bottom) */}
                <AdvancedPipe x1={650} y1={245} x2={690} y2={245} isActive={electricFlowing} />

                {/* Junction Dot (Glass Style) */}
                <circle cx="650" cy="185" r="6" fill="#1e293b" stroke={electricFlowing ? glowCyan : dimColor} strokeWidth="2" />
                <circle cx="650" cy="185" r="2" fill="white" />

                {/* 锂电池 (统一玻璃风格) */}
                {/* 下移 25px (60 -> 85) */}
                <g transform="translate(690, 85)">
                    <rect x="0" y="0" width="85" height="80" rx="6"
                        fill="rgba(0, 0, 0, 0.5)"
                        stroke={isRunning ? '#10B981' : 'rgba(255,255,255,0.1)'}
                        strokeWidth={isRunning ? 2 : 1}
                        filter={isRunning ? 'url(#glow)' : 'none'} />
                    <rect x="32" y="-8" width="20" height="10" rx="2" fill={isRunning ? '#10B981' : 'rgba(16, 185, 129, 0.3)'} />
                    <text x="42" y="30" textAnchor="middle" fill="#10B981" fontSize={FONT_SIZES.labelMedium} fontWeight="bold">锂电池</text>
                    <text x="42" y="55" textAnchor="middle" fontSize={FONT_SIZES.icon}>🔋</text>
                </g>

                {/* 电子负载 (统一玻璃风格) */}
                {/* 下移 25px (180 -> 205) */}
                <g transform="translate(690, 205)">
                    <rect x="0" y="0" width="85" height="80" rx="6"
                        fill="rgba(0, 0, 0, 0.5)"
                        stroke={electricFlowing ? '#8B5CF6' : 'rgba(255,255,255,0.1)'}
                        strokeWidth={electricFlowing ? 2 : 1}
                        filter={electricFlowing ? 'url(#glow)' : 'none'} />
                    <text x="42" y="35" textAnchor="middle" fill="#A78BFA" fontSize={FONT_SIZES.labelMedium} fontWeight="bold">电子负载</text>
                    <text x="42" y="55" textAnchor="middle" fill="#8B5CF6" fontSize={FONT_SIZES.labelMedium - 2}>DCL</text>
                </g>

                {/* 风扇2 (散热) - 带玻璃罩 */}
                {/* 下移 25px (276 -> 301) */}
                <g transform="translate(525, 301)">
                    {/* Glass Background */}
                    <rect x="-10" y="-10" width="60" height="70" rx="4" fill="rgba(0,0,0,0.3)" stroke="none" />

                    <circle cx="20" cy="20" r="16" fill="url(#glassGradient)" stroke={data.io.fan2 ? glowCyan : dimColor} strokeWidth="2" />
                    <path d="M20 20 L20 6 M20 20 L32 28 M20 20 L8 28" stroke={data.io.fan2 ? glowCyan : dimColor} strokeWidth="2"
                        className={data.io.fan2 ? "origin-[20px_20px] animate-spin" : ""} />
                    <text x="20" y="55" textAnchor="middle" fill="#90CAF9" fontSize={FONT_SIZES.labelSmall}>风扇2(散热)</text>
                </g>
                {/* 替换为 AdvancedPipe 连接线 */}
                {/* DCF Bottom 280. (+25 from 255). Pipe end (+25 from 280 -> 305). */}
                <AdvancedPipe x1={545} y1={280} x2={545} y2={305} isActive={data.io.fan2} />

            </svg>
        </div >
    );
};
