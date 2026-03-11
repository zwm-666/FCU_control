import { useState, useEffect, useCallback, type ElementType } from "react";
import {
  INITIAL_MACHINE_STATE,
  INITIAL_CONTROL_STATE,
  MachineState,
  ControlState,
  SystemState,
  FAULT_CODES,
  ConnectionConfig,
  DiagnosisLabel,
  DiagnosisResult,
} from "./types";
import { generateControlPacket } from "./services/canProtocol";
import { wsService } from "./services/websocketService";

// Custom hooks
import { useDataLogger } from "./hooks/useDataLogger";
import { useChartHistory } from "./hooks/useChartHistory";
import { useFaultLogs } from "./hooks/useFaultLogs";

// Shared components
import { AlarmDrawer } from "./components/AlarmDrawer";
import { GlassPanel } from "./components/GlassPanel";
import { MetricCard } from "./components/MetricCard";
import { DiagnosisPanel } from "./components/DiagnosisPanel";
import { BottomControlPanel } from "./components/BottomControlPanel";
import { ConfirmationModal, ModalType } from "./components/ConfirmationModal";

// View components
import { MonitorView } from "./components/views/MonitorView";
import { ChartsView } from "./components/views/ChartsView";
import { ControlView } from "./components/views/ControlView";
import { AlarmsView } from "./components/views/AlarmsView";

// Icons
import {
  Wifi,
  WifiOff,
  Save,
  Square,
  Activity,
  BarChart2,
  Settings,
  AlertTriangle,
  Zap,
  Thermometer,
  Gauge,
  RotateCcw,
  Bell,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types & constants
// ---------------------------------------------------------------------------

type ViewType = "monitor" | "charts" | "control" | "alarms";

const NAV_TABS: { id: ViewType; icon: ElementType; label: string }[] = [
  { id: "monitor", icon: Activity, label: "监控" },
  { id: "charts", icon: BarChart2, label: "图表" },
  { id: "control", icon: Settings, label: "设置" },
  { id: "alarms", icon: AlertTriangle, label: "报警" },
];

const STATUS_TEXT: Record<SystemState, string> = {
  [SystemState.OFF]: "待机",
  [SystemState.START]: "启动中",
  [SystemState.RUN]: "运行中",
  [SystemState.FAULT]: "故障",
};

const STATUS_DOT: Record<SystemState, string> = {
  [SystemState.OFF]: "bg-slate-400",
  [SystemState.START]: "bg-amber-400",
  [SystemState.RUN]: "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.5)]",
  [SystemState.FAULT]: "bg-rose-400 shadow-[0_0_10px_rgba(251,113,133,0.5)]",
};

// ---------------------------------------------------------------------------
// Modal state shape
// ---------------------------------------------------------------------------
interface ModalConfig {
  isOpen: boolean;
  title: string;
  message: string;
  type: ModalType;
  onConfirm: () => void;
}

const CLOSED_MODAL: ModalConfig = {
  isOpen: false,
  title: "",
  message: "",
  type: "info",
  onConfirm: () => {},
};

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  // Core machine / control state
  const [machine, setMachine] = useState<MachineState>(INITIAL_MACHINE_STATE);
  const [control, setControl] = useState<ControlState>(INITIAL_CONTROL_STATE);

  // Connection lifecycle
  const [isConnected, setIsConnected] = useState(false);
  const [isSystemRunning, setIsSystemRunning] = useState(false);

  // UI state
  const [activeView, setActiveView] = useState<ViewType>("monitor");
  const [isAlarmDrawerOpen, setIsAlarmDrawerOpen] = useState(false);
  const [modalConfig, setModalConfig] = useState<ModalConfig>(CLOSED_MODAL);

  // CAN / connection config
  const [connectionConfig, setConnectionConfig] = useState<ConnectionConfig>({
    interfaceType: "virtual",
    channel: "can0",
    bitrate: "250000",
  });

  // Diagnosis result (streamed from backend)
  const [diagnosis, setDiagnosis] = useState<DiagnosisResult | null>(null);

  // Clock
  const [currentTime, setCurrentTime] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // ---------------------------------------------------------------------------
  // Custom hooks
  // ---------------------------------------------------------------------------
  const { isLogging, handleToggleLog } = useDataLogger(machine);
  const { history } = useChartHistory(isConnected, machine);
  const { faultLogs } = useFaultLogs(
    machine.io.faultCode,
    machine.status.faultLevel,
  );

  // ---------------------------------------------------------------------------
  // WebSocket lifecycle
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (isSystemRunning) {
      setIsConnected(true);
      wsService.connect("ws://localhost:8765");

      const unsubState = wsService.onMachineState((s) => setMachine(s));
      const unsubConnection = wsService.onConnection((connected) => {
        if (!connected) setMachine((prev) => ({ ...prev, connected: false }));
      });
      const unsubDiagnosis = wsService.onDiagnosis((r) => setDiagnosis(r));

      return () => {
        unsubState();
        unsubConnection();
        unsubDiagnosis();
      };
    } else {
      setIsConnected(false);
      wsService.disconnect();
      setMachine(INITIAL_MACHINE_STATE);
      setDiagnosis(null);
    }
  }, [isSystemRunning]);

  // ---------------------------------------------------------------------------
  // Control update (TX)
  // ---------------------------------------------------------------------------
  const handleControlUpdate = useCallback(
    (updates: Partial<ControlState>) => {
      const hasChanged = Object.keys(updates).some(
        (k) =>
          control[k as keyof ControlState] !== updates[k as keyof ControlState],
      );
      if (!hasChanged) return;

      const next = { ...control, ...updates };
      setControl(next);
      wsService.sendControl(next);

      if (import.meta.env.DEV) {
        const packet = generateControlPacket(next);
        console.log(
          "[TX] CAN ID:",
          packet.id.toString(16),
          "DATA:",
          packet.data,
        );
      }
    },
    [control],
  );

  // ---------------------------------------------------------------------------
  // Diagnosis feedback
  // ---------------------------------------------------------------------------
  const handleDiagnosisFeedback = useCallback((label: DiagnosisLabel) => {
    wsService.sendDiagnosisFeedback(label);
  }, []);

  // ---------------------------------------------------------------------------
  // Modal helpers
  // ---------------------------------------------------------------------------
  const showModal = useCallback(
    (
      title: string,
      message: string,
      type: ModalType,
      onConfirm: () => void,
    ) => {
      setModalConfig({ isOpen: true, title, message, type, onConfirm });
    },
    [],
  );

  const closeModal = useCallback(() => setModalConfig(CLOSED_MODAL), []);

  // Toggle system running with modal confirmation on stop
  const handleToggleSystem = useCallback(() => {
    if (isSystemRunning) {
      showModal(
        "停止系统连接",
        "确认停止系统？这将断开与燃料电池的 WebSocket 连接。",
        "warning",
        () => setIsSystemRunning(false),
      );
    } else {
      setIsSystemRunning(true);
    }
  }, [isSystemRunning, showModal]);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  const systemState = machine.status.state;
  const activeFaultCount = faultLogs.length;

  return (
    <div className="h-screen w-screen flex flex-col bg-[#020617] text-slate-100 font-sans overflow-hidden relative">
      {/* Ambient glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[80vw] h-[300px] bg-cyan-500/10 blur-[100px] rounded-full pointer-events-none" />

      {/* ===== HEADER ===== */}
      <header className="shrink-0 z-50 px-6 py-2 flex justify-between items-center bg-slate-900/80 backdrop-blur-md border-b border-white/5 shadow-xl relative">
        {/* Logo + title */}
        <div className="flex items-center gap-4">
          <div className="relative">
            <div className="absolute inset-0 bg-cyan-400/20 blur-lg rounded-full" />
            <div className="relative border border-cyan-500/30 bg-black/40 backdrop-blur-md p-2 rounded-lg">
              <span className="text-cyan-400 text-2xl font-bold">◈</span>
            </div>
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">
              氢燃料电池监控系统
            </h1>
            <div className="flex items-center gap-2 text-xs font-mono text-cyan-500/80">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              实时在线监控
            </div>
          </div>
        </div>

        {/* Center nav tabs */}
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
          <div className="flex bg-slate-800/50 backdrop-blur-md rounded-full p-1 border border-white/5">
            {NAV_TABS.map(({ id, icon: Icon, label }) => (
              <button
                key={id}
                onClick={() => setActiveView(id)}
                className={`px-4 py-1.5 rounded-full text-xs font-medium transition-all flex items-center gap-2 ${
                  activeView === id
                    ? "bg-cyan-500 text-slate-900 shadow-lg shadow-cyan-500/20"
                    : "text-slate-400 hover:text-white hover:bg-white/5"
                }`}
              >
                <Icon className="w-3 h-3" />
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Right controls */}
        <div className="flex items-center gap-4">
          {/* System status badge */}
          <GlassPanel className="!p-1.5 px-3 flex items-center gap-3 !bg-slate-800/50">
            <div
              className={`w-2.5 h-2.5 rounded-full animate-pulse ${STATUS_DOT[systemState]}`}
            />
            <span className="text-xs font-bold text-slate-200">
              {STATUS_TEXT[systemState] ?? "未知"}
            </span>
          </GlassPanel>

          {/* Clock */}
          <div className="flex flex-col items-end">
            <span className="text-xl font-mono text-cyan-400 tracking-wide font-bold leading-none">
              {currentTime.toLocaleTimeString("zh-CN", { hour12: false })}
            </span>
            <span className="text-[10px] font-mono text-slate-400 tracking-wider uppercase mt-1">
              {currentTime.toLocaleDateString("zh-CN")}
            </span>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-2 border-l border-white/10 pl-4">
            {/* Alarm drawer toggle */}
            <button
              onClick={() => setIsAlarmDrawerOpen(true)}
              className={`relative p-2 rounded-lg transition-all ${
                activeFaultCount > 0
                  ? "bg-rose-500/20 text-rose-400 shadow-[0_0_10px_rgba(244,63,94,0.2)]"
                  : "bg-white/5 text-slate-500 hover:text-slate-300"
              }`}
              title="报警记录"
            >
              <Bell className="w-4 h-4" />
              {activeFaultCount > 0 && (
                <span className="absolute -top-1 -right-1 flex h-4 w-4 items-center justify-center rounded-full bg-rose-500 text-[9px] font-bold text-white">
                  {activeFaultCount > 9 ? "9+" : activeFaultCount}
                </span>
              )}
            </button>

            {/* CSV log toggle */}
            <button
              onClick={handleToggleLog}
              className={`p-2 rounded-lg transition-all ${
                isLogging
                  ? "bg-rose-500/20 text-rose-400 shadow-[0_0_10px_rgba(244,63,94,0.2)]"
                  : "bg-white/5 text-slate-500 hover:text-slate-300"
              }`}
              title={isLogging ? "停止记录" : "开始数据记录"}
            >
              {isLogging ? (
                <Square className="w-4 h-4 fill-current animate-pulse" />
              ) : (
                <Save className="w-4 h-4" />
              )}
            </button>

            {/* System connect / disconnect */}
            <button
              onClick={handleToggleSystem}
              className={`p-2 rounded-lg transition-all ${
                isSystemRunning
                  ? "bg-cyan-500/20 text-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.2)]"
                  : "bg-white/5 text-slate-500 hover:text-slate-300"
              }`}
              title={isSystemRunning ? "断开连接" : "连接系统"}
            >
              {isSystemRunning ? (
                <Wifi className="w-4 h-4" />
              ) : (
                <WifiOff className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>
      </header>

      {/* ===== MAIN GRID ===== */}
      <main className="flex-1 min-h-0 p-4 grid grid-cols-12 grid-rows-[1fr_auto] gap-2 relative z-0">
        {/* LEFT: Vital stats + diagnosis (3 cols) */}
        <div className="col-span-3 flex flex-col gap-4 h-full min-h-0">
          <div className="flex-1 overflow-y-auto no-scrollbar space-y-2 pr-1">
            <GlassPanel
              title="电堆核心指标"
              icon={<Activity className="w-3.5 h-3.5" />}
              className="shrink-0"
              contentClassName="p-3"
            >
              <div className="space-y-2">
                <MetricCard
                  label="电堆电压"
                  value={machine.power.stackVoltage.toFixed(1)}
                  unit="V"
                  icon={<Zap className="w-4 h-4" />}
                  color="cyan"
                  subValue="目标值: ~48V"
                />
                <MetricCard
                  label="电堆电流"
                  value={machine.power.stackCurrent.toFixed(1)}
                  unit="A"
                  icon={<Zap className="w-4 h-4" />}
                  color="blue"
                />
                <MetricCard
                  label="电堆温度"
                  value={machine.sensors.stackTemp.toFixed(1)}
                  unit="°C"
                  icon={<Thermometer className="w-4 h-4" />}
                  color="amber"
                  subValue="告警阈值: 75°C"
                />
              </div>
            </GlassPanel>

            <GlassPanel
              title="智能故障诊断"
              icon={<RotateCcw className="w-3.5 h-3.5" />}
              className="shrink-0"
              contentClassName="p-3"
            >
              <DiagnosisPanel
                diagnosis={diagnosis}
                onFeedback={handleDiagnosisFeedback}
                lastFaultCode={machine.io.faultCode}
              />
            </GlassPanel>
          </div>
        </div>

        {/* CENTER: Dynamic view (6 cols) */}
        <div className="col-span-6 relative flex flex-col h-full min-h-0 shadow-2xl rounded-2xl bg-slate-900/20 border border-white/5 backdrop-blur-sm overflow-hidden">
          <div className="flex-1 relative overflow-auto custom-scrollbar">
            {activeView === "monitor" && <MonitorView data={machine} />}
            {activeView === "charts" && <ChartsView history={history} />}
            {activeView === "control" && (
              <ControlView
                connectionConfig={connectionConfig}
                onConfigChange={(updates: Partial<ConnectionConfig>) =>
                  setConnectionConfig((prev) => ({ ...prev, ...updates }))
                }
              />
            )}
            {activeView === "alarms" && <AlarmsView faultLogs={faultLogs} />}
          </div>
        </div>

        {/* RIGHT: Pressure + DCF (3 cols) */}
        <div className="col-span-3 flex flex-col gap-4 h-full min-h-0">
          <div className="flex-1 overflow-y-auto no-scrollbar space-y-2 pr-1">
            <GlassPanel
              title="系统压力与DCF"
              icon={<Gauge className="w-3.5 h-3.5" />}
              className="shrink-0"
              contentClassName="p-3"
            >
              <div className="grid grid-cols-1 gap-3">
                <MetricCard
                  label="氢瓶压力"
                  value={machine.sensors.h2CylinderPressure.toFixed(2)}
                  unit="MPa"
                  color="cyan"
                />
                <MetricCard
                  label="进氢压力"
                  value={machine.sensors.h2InletPressure.toFixed(2)}
                  unit="MPa"
                  color="emerald"
                />
                <MetricCard
                  label="氢气浓度"
                  value={machine.sensors.h2Concentration.toFixed(1)}
                  unit="%"
                  color={
                    machine.sensors.h2Concentration > 1.0 ? "rose" : "blue"
                  }
                  subValue="告警阈值: 2.0%"
                />

                {/* DCF output detail */}
                <div className="p-3 bg-slate-900/50 rounded-lg border border-white/5">
                  <span className="text-xs text-slate-400 block mb-2">
                    DCF 输出状态
                  </span>

                  {[
                    {
                      label: "输出电压",
                      value: machine.power.dcfOutVoltage,
                      unit: "V",
                      max: 60,
                      barColor: "bg-cyan-500",
                      textColor: "text-cyan-400",
                    },
                    {
                      label: "输出电流",
                      value: machine.power.dcfOutCurrent,
                      unit: "A",
                      max: 100,
                      barColor: "bg-blue-500",
                      textColor: "text-cyan-400",
                    },
                    {
                      label: "DCF温度",
                      value: machine.io.dcfMosTemp,
                      unit: "℃",
                      max: 100,
                      barColor:
                        machine.io.dcfMosTemp > 60
                          ? "bg-rose-500"
                          : "bg-amber-500",
                      textColor:
                        machine.io.dcfMosTemp > 60
                          ? "text-rose-400 animate-pulse"
                          : "text-amber-400",
                    },
                  ].map(({ label, value, unit, max, barColor, textColor }) => (
                    <div key={label} className="mb-3 last:mb-0">
                      <div className="flex justify-between items-end mb-1">
                        <span className="text-slate-300 text-sm">{label}</span>
                        <span className={`font-mono text-sm ${textColor}`}>
                          {value.toFixed(1)} {unit}
                        </span>
                      </div>
                      <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${barColor} transition-all duration-300`}
                          style={{
                            width: `${Math.min(100, (value / max) * 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </GlassPanel>
          </div>
        </div>

        {/* BOTTOM: Control panel — only on monitor view */}
        {activeView === "monitor" && (
          <div className="col-span-12 h-auto flex-none">
            <BottomControlPanel
              control={control}
              onUpdate={handleControlUpdate}
            />
          </div>
        )}
      </main>

      {/* ===== FOOTER ===== */}
      <footer className="h-8 bg-black/60 backdrop-blur-md border-t border-white/10 flex items-center justify-between px-6 text-[10px] text-slate-400 font-mono z-20 relative">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-500/50" />
            CAN 总线通信{isConnected ? "已连接" : "就绪"}
          </span>
          <span className="text-slate-700">|</span>
          <span>Bitrate: {connectionConfig.bitrate} bps</span>
        </div>

        {/* Fault ticker */}
        <div className="flex-1 mx-8 relative h-full overflow-hidden flex items-center justify-center">
          {machine.io.faultCode !== 0 && (
            <div className="text-rose-400 font-bold animate-pulse whitespace-nowrap">
              ⚠ 系统告警:{" "}
              {FAULT_CODES[machine.io.faultCode] ??
                `未知故障代码 0x${machine.io.faultCode.toString(16).toUpperCase()}`}{" "}
              ⚠
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span>H2-FCU v1.2.0</span>
          <span className="text-slate-700">|</span>
          <span className="text-cyan-500">ANTIGRAVITY DESIGN</span>
        </div>
      </footer>

      {/* ===== ALARM DRAWER ===== */}
      <AlarmDrawer
        isOpen={isAlarmDrawerOpen}
        onClose={() => setIsAlarmDrawerOpen(false)}
        logs={faultLogs}
      />

      {/* ===== CONFIRMATION MODAL ===== */}
      <ConfirmationModal
        isOpen={modalConfig.isOpen}
        title={modalConfig.title}
        message={modalConfig.message}
        type={modalConfig.type}
        onConfirm={() => {
          modalConfig.onConfirm();
          closeModal();
        }}
        onCancel={closeModal}
      />
    </div>
  );
}

export default App;
