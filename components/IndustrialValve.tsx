import React from 'react';

interface Props {
    x: number;
    y: number;
    isOpen: boolean;
    orientation?: 'horizontal' | 'vertical'; // Default horizontal
    label?: string;
    scale?: number;
}

export const IndustrialValve: React.FC<Props> = ({
    x,
    y,
    isOpen,
    orientation = 'horizontal',
    label,
    scale = 1
}) => {
    const glowGreen = '#00FF88';
    const dimColor = '#3a5070';
    const activeColor = isOpen ? glowGreen : dimColor;

    // Adjust rotation based on orientation
    const rotation = orientation === 'vertical' ? 90 : 0;

    return (
        <g transform={`translate(${x}, ${y}) scale(${scale})`}>
            {/* Hit Area / Background for clickability if needed later */}
            <rect x="-20" y="-30" width="40" height="60" fill="transparent" stroke="none" />

            {/* Valve Logic Container - Rotated */}
            <g transform={`rotate(${rotation})`}>
                {/* 1. Main Body: Two opposing triangles (Bow-tie) */}
                {/* Points for horizontal bow-tie: Left(-18, -10/10) to Center(0,0) to Right(18, -10/10) */}
                {/* Using a path to draw the bow-tie shape */}
                <path d="M -15 -10 L 0 0 L -15 10 Z  M 15 -10 L 0 0 L 15 10 Z"
                    fill="none"
                    stroke={activeColor}
                    strokeWidth="2"
                    filter={isOpen ? 'url(#glow)' : 'none'}
                />

                {/* 2. Center Pivot Point replaced by Line? User said: "valve on the circle changed to -" */}
                {/* "相交的地方有一个t管吗" -> A T-shaped intersection? */}
                {/* Usually ISO valves have a dot. User might mean the stem connection looks like a T. */}
                {/* "Changed the circle ON THE VALVE to a -" -> Maybe the center dot becomes a small vertical line? */}
                {/* Or perhaps the user meant the Actuator Circle? */}
                {/* Let's assume Actuator Circle -> Handle Line. */}
                {/* But "相交的地方" (Intersection) usually refers to the center. */}
                {/* Let's Draw a line across the center? */}

                {/* Let's interpret: Center Pivot is a connection point. */}
                <circle cx="0" cy="0" r="2" fill={activeColor} />

                {/* 3. Stem (Connection to Actuator) */}
                {/* Stem goes UP relative to the valve body */}
                <line x1="0" y1="0" x2="0" y2="-20" stroke={activeColor} strokeWidth="2" />

                {/* 4. Actuator (Handle) */}
                {/* User: "change the circle to a -" -> A T-bar handle for manual valve */}
                {/* Draw a horizontal line at the top of the stem */}
                <line x1="-10" y1="-20" x2="10" y2="-20" stroke={activeColor} strokeWidth="2" />
            </g>

            {/* Label - Always upright, so we counter-rotate or place relative to group */}
            {label && (
                <g>
                    <text x="0" y={orientation === 'vertical' ? 45 : 35}
                        textAnchor="middle"
                        fill="#64B5F6"
                        fontSize="9" // Scale sensitive? Maybe fixed size
                        fontWeight="bold"
                        style={{ textShadow: '0 1px 2px rgba(0,0,0,0.8)' }}
                    >
                        {label}
                    </text>
                    <text x="0" y={orientation === 'vertical' ? 58 : 48}
                        textAnchor="middle"
                        fill={isOpen ? glowGreen : '#607D8B'}
                        fontSize="8"
                    >
                        ({isOpen ? '开启' : '关闭'})
                    </text>
                </g>
            )}
        </g>
    );
};
