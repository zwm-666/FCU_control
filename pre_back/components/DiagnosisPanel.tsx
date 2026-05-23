import * as React from "react";
import {
  DiagnosisResult,
  DiagnosisLabel,
  DIAGNOSIS_LABELS_CN,
  FAULT_CODES,
} from "../types";
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Droplets,
  Thermometer,
  HelpCircle,
} from "lucide-react";

interface Props {
  diagnosis: DiagnosisResult | null;
  onFeedback: (label: DiagnosisLabel) => void;
  lastFaultCode: number;
}

export const DiagnosisPanel: React.FC<Props> = ({
  diagnosis,
  onFeedback,
  lastFaultCode,
}: Props) => {
  if (!diagnosis) {
    return (
      <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700">
        <div className="flex items-center gap-2 text-slate-400">
          <HelpCircle className="w-5 h-5" />
          <span className="text-sm">诊断模块未连接</span>
        </div>
      </div>
    );
  }

  const getLabelStyle = (label: DiagnosisLabel) => {
    switch (label) {
      case "normal":
        return {
          bg: "bg-emerald-500/20",
          border: "border-emerald-500/50",
          text: "text-emerald-400",
          icon: <CheckCircle className="w-6 h-6" />,
        };
      case "flooding":
        return {
          bg: "bg-blue-500/20",
          border: "border-blue-500/50",
          text: "text-blue-400",
          icon: <Droplets className="w-6 h-6" />,
        };
      case "membrane_drying":
        return {
          bg: "bg-amber-500/20",
          border: "border-amber-500/50",
          text: "text-amber-400",
          icon: <AlertTriangle className="w-6 h-6" />,
        };
      case "thermal_issue":
        return {
          bg: "bg-red-500/20",
          border: "border-red-500/50",
          text: "text-red-400",
          icon: <Thermometer className="w-6 h-6" />,
        };
      default:
        return {
          bg: "bg-slate-500/20",
          border: "border-slate-500/50",
          text: "text-slate-400",
          icon: <Activity className="w-6 h-6" />,
        };
    }
  };

  const style = getLabelStyle(diagnosis.label);

  // Show a mismatch warning when the ML model reports "normal" but there is an active fault code
  const hasFaultMismatch =
    lastFaultCode !== 0 &&
    diagnosis.label === "normal" &&
    diagnosis.confidence > 70;

  const CircularProgress = ({ value }: { value: number }) => {
    const radius = 36;
    const circumference = 2 * Math.PI * radius;
    const progress = (value / 100) * circumference;

    const strokeColor =
      diagnosis.label === "normal"
        ? "#10B981"
        : diagnosis.label === "flooding"
          ? "#3B82F6"
          : diagnosis.label === "membrane_drying"
            ? "#F59E0B"
            : "#EF4444";

    return (
      <div className="relative w-24 h-24">
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="48"
            cy="48"
            r={radius}
            fill="none"
            stroke="#334155"
            strokeWidth="6"
          />
          <circle
            cx="48"
            cy="48"
            r={radius}
            fill="none"
            stroke={strokeColor}
            strokeWidth="6"
            strokeDasharray={circumference}
            strokeDashoffset={circumference - progress}
            strokeLinecap="round"
            className="transition-all duration-500"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-xl font-bold font-mono ${style.text}`}>
            {value.toFixed(0)}%
          </span>
          <span className="text-[10px] text-slate-500">置信度</span>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-slate-900/80 rounded-lg border border-slate-700/50 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 px-4 py-2 flex items-center justify-between border-b border-slate-600/50">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span className="text-sm font-bold text-slate-100">实时诊断</span>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-slate-500">
          <span>样本: {diagnosis.sample_count}</span>
          {diagnosis.is_trained ? (
            <span className="px-1.5 py-0.5 bg-emerald-500/20 text-emerald-400 rounded">
              已训练
            </span>
          ) : (
            <span className="px-1.5 py-0.5 bg-amber-500/20 text-amber-400 rounded">
              未训练
            </span>
          )}
        </div>
      </div>

      {/* Active fault code context banner */}
      {lastFaultCode !== 0 && (
        <div className="px-4 py-1.5 bg-rose-900/30 border-b border-rose-500/20 flex items-center gap-2">
          <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
          <span className="text-[10px] text-rose-300 font-mono">
            当前故障码: 0x
            {lastFaultCode.toString(16).toUpperCase().padStart(2, "0")}
            {" — "}
            {FAULT_CODES[lastFaultCode] ?? "未知故障"}
          </span>
        </div>
      )}

      {/* Mismatch warning */}
      {hasFaultMismatch && (
        <div className="px-4 py-1.5 bg-amber-900/30 border-b border-amber-500/20 flex items-center gap-2">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          <span className="text-[10px] text-amber-300">
            模型判断正常，但存在活跃故障码，请人工核查
          </span>
        </div>
      )}

      {/* Diagnosis result */}
      <div className="p-4">
        <div className="flex items-center gap-4">
          <CircularProgress value={diagnosis.confidence} />

          <div className="flex-1">
            <div
              className={`inline-flex items-center gap-2 px-3 py-2 rounded-lg ${style.bg} ${style.border} border`}
            >
              <span className={style.text}>{style.icon}</span>
              <span className={`font-bold ${style.text}`}>
                {diagnosis.label_cn}
              </span>
            </div>

            {diagnosis.probabilities &&
              Object.keys(diagnosis.probabilities).length > 0 && (
                <div className="mt-3 space-y-1">
                  {Object.entries(diagnosis.probabilities).map(
                    ([label, prob]) => (
                      <div
                        key={label}
                        className="flex items-center gap-2 text-[10px]"
                      >
                        <span className="w-16 text-slate-500 truncate">
                          {DIAGNOSIS_LABELS_CN[label as DiagnosisLabel] ??
                            label}
                        </span>
                        <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${
                              label === "normal"
                                ? "bg-emerald-500"
                                : label === "flooding"
                                  ? "bg-blue-500"
                                  : label === "membrane_drying"
                                    ? "bg-amber-500"
                                    : "bg-red-500"
                            }`}
                            style={{ width: `${prob}%` }}
                          />
                        </div>
                        <span className="w-10 text-right text-slate-400 font-mono">
                          {(prob as number).toFixed(1)}%
                        </span>
                      </div>
                    ),
                  )}
                </div>
              )}
          </div>
        </div>

        {/* Feedback buttons */}
        <div className="mt-4 pt-3 border-t border-slate-700/50">
          <div className="text-[10px] text-slate-500 mb-2">
            标注反馈 (用于模型训练)
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => onFeedback("normal")}
              className="px-2 py-1 text-[10px] rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/20 transition-colors"
            >
              ✓ 正常
            </button>
            <button
              onClick={() => onFeedback("flooding")}
              className="px-2 py-1 text-[10px] rounded bg-blue-500/10 text-blue-400 border border-blue-500/30 hover:bg-blue-500/20 transition-colors"
            >
              💧 水淹
            </button>
            <button
              onClick={() => onFeedback("membrane_drying")}
              className="px-2 py-1 text-[10px] rounded bg-amber-500/10 text-amber-400 border border-amber-500/30 hover:bg-amber-500/20 transition-colors"
            >
              ⚠ 膜干燥
            </button>
            <button
              onClick={() => onFeedback("thermal_issue")}
              className="px-2 py-1 text-[10px] rounded bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20 transition-colors"
            >
              🔥 热管理
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
