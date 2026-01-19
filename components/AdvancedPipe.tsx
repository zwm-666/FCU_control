import React from 'react';

/**
 * AdvancedPipe - 玻璃管道组件
 * 
 * 设计：类似玻璃管的效果，蓝/绿色，开启后有流动效果
 * - 底层：半透明玻璃管外壳
 * - 中层：发光核心
 * - 顶层：高光流动动画
 */

interface AdvancedPipeProps {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    /** 是否激活（控制发光和流动效果） */
    isActive: boolean;
}

// 统一蓝绿色
const GLASS_COLOR = '#00E5FF';      // 亮青色
const GLASS_DIM = '#1a4a5a';        // 暗淡青色
const HIGHLIGHT_COLOR = '#e0ffff';  // 极亮的青白

export const AdvancedPipe: React.FC<AdvancedPipeProps> = ({
    x1, y1, x2, y2,
    isActive
}) => {
    const glassColor = isActive ? GLASS_COLOR : GLASS_DIM;

    return (
        <g className="glass-pipe">
            {/* ========== 底层 - 玻璃管外壳 ========== */}
            {/* 半透明，带模糊效果，模拟玻璃管 */}
            <line
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke={glassColor}
                strokeWidth={14}
                strokeLinecap="round"
                opacity={isActive ? 0.25 : 0.15}
                filter="url(#heavy-blur)"
            />

            {/* ========== 中层 - 玻璃管壁 ========== */}
            {/* 实体管壁，半透明 */}
            <line
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke={glassColor}
                strokeWidth={8}
                strokeLinecap="round"
                opacity={isActive ? 0.6 : 0.3}
            />

            {/* ========== 顶层 - 高光流动层 ========== */}
            {/* 中心亮线，带流动动画 */}
            <line
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke={isActive ? HIGHLIGHT_COLOR : GLASS_DIM}
                strokeWidth={3}
                strokeLinecap="round"
                strokeDasharray={isActive ? "15 25" : "0"}
                className={isActive ? "pipe-flow-animation" : ""}
                opacity={isActive ? 0.9 : 0.2}
            />
        </g>
    );
};
