/**
 * WebSocket Service for CAN Data Communication
 * Connects to Python backend WebSocket server
 */

import {
  MachineState,
  ControlState,
  DiagnosisResult,
  DiagnosisLabel,
} from "../types";

type MachineStateCallback = (state: MachineState) => void;
type ConnectionCallback = (connected: boolean) => void;
type DiagnosisCallback = (result: DiagnosisResult) => void;

const isDev = import.meta.env.DEV;

// Reconnect configuration
const RECONNECT_BASE_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 30000;
const RECONNECT_MAX_ATTEMPTS = 10;
const RECONNECT_JITTER_MS = 500;

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectTimer: number | null = null;
  private reconnectUrl: string | null = null;

  private machineStateCallbacks: Set<MachineStateCallback> = new Set();
  private connectionCallbacks: Set<ConnectionCallback> = new Set();
  private diagnosisCallbacks: Set<DiagnosisCallback> = new Set();

  private reconnectAttempts = 0;
  /** Whether the user intentionally disconnected — suppresses auto-reconnect */
  private intentionalDisconnect = false;

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  connect(url: string = "ws://localhost:8765"): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.intentionalDisconnect = false;
    this.reconnectUrl = url;

    if (isDev) console.log(`[WS] Connecting to ${url}`);
    this.createSocket(url);
  }

  disconnect(): void {
    this.intentionalDisconnect = true;
    this.clearReconnectTimer();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.reconnectAttempts = 0;
    this.notifyConnection(false);
  }

  sendControl(control: ControlState): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;

    this.ws.send(JSON.stringify({ type: "control", data: control }));
    if (isDev) console.log("[WS] TX Control:", control);
  }

  sendDiagnosisFeedback(label: DiagnosisLabel): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;

    this.ws.send(
      JSON.stringify({ type: "diagnosis_feedback", data: { label } }),
    );
    if (isDev) console.log("[WS] TX Diagnosis Feedback:", label);
  }

  onMachineState(callback: MachineStateCallback): () => void {
    this.machineStateCallbacks.add(callback);
    return () => {
      this.machineStateCallbacks.delete(callback);
    };
  }

  onConnection(callback: ConnectionCallback): () => void {
    this.connectionCallbacks.add(callback);
    // Immediately deliver the current connection state to the new subscriber
    callback(this.ws?.readyState === WebSocket.OPEN);
    return () => {
      this.connectionCallbacks.delete(callback);
    };
  }

  onDiagnosis(callback: DiagnosisCallback): () => void {
    this.diagnosisCallbacks.add(callback);
    return () => {
      this.diagnosisCallbacks.delete(callback);
    };
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  private createSocket(url: string): void {
    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        if (isDev) console.log("[WS] Connected");
        this.reconnectAttempts = 0;
        this.notifyConnection(true);
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };

      this.ws.onerror = (error) => {
        console.error("[WS] Error:", error);
      };

      this.ws.onclose = () => {
        if (isDev) console.log("[WS] Disconnected");
        this.notifyConnection(false);

        if (!this.intentionalDisconnect) {
          this.scheduleReconnect();
        }
      };
    } catch (error) {
      console.error("[WS] Failed to create WebSocket:", error);
      this.notifyConnection(false);
      if (!this.intentionalDisconnect) {
        this.scheduleReconnect();
      }
    }
  }

  /**
   * Exponential backoff with jitter:
   *   delay = min(base * 2^attempt, maxDelay) + random(0, jitter)
   *
   * attempt 0 →  1.0 s + jitter
   * attempt 1 →  2.0 s + jitter
   * attempt 2 →  4.0 s + jitter
   * attempt 3 →  8.0 s + jitter
   * attempt 4 → 16.0 s + jitter
   * attempt 5 → 30.0 s + jitter  (capped)
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= RECONNECT_MAX_ATTEMPTS) {
      console.error("[WS] Max reconnection attempts reached. Giving up.");
      return;
    }

    const exponentialDelay = Math.min(
      RECONNECT_BASE_DELAY_MS * Math.pow(2, this.reconnectAttempts),
      RECONNECT_MAX_DELAY_MS,
    );
    const jitter = Math.random() * RECONNECT_JITTER_MS;
    const delay = exponentialDelay + jitter;

    this.reconnectAttempts++;

    if (isDev) {
      console.log(
        `[WS] Reconnecting in ${(delay / 1000).toFixed(1)}s ` +
          `(attempt ${this.reconnectAttempts}/${RECONNECT_MAX_ATTEMPTS})`,
      );
    }

    this.reconnectTimer = window.setTimeout(() => {
      if (!this.intentionalDisconnect && this.reconnectUrl) {
        this.createSocket(this.reconnectUrl);
      }
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private handleMessage(data: string): void {
    try {
      const message = JSON.parse(data);

      if (message.type === "machine_state") {
        this.notifyMachineState(message.data as MachineState);

        if (message.diagnosis) {
          this.notifyDiagnosis(message.diagnosis as DiagnosisResult);
        }
      }
    } catch (error) {
      console.error("[WS] Failed to parse message:", error);
    }
  }

  private notifyMachineState(state: MachineState): void {
    this.machineStateCallbacks.forEach((cb) => {
      try {
        cb(state);
      } catch (e) {
        console.error("[WS] machineState callback error:", e);
      }
    });
  }

  private notifyConnection(connected: boolean): void {
    this.connectionCallbacks.forEach((cb) => {
      try {
        cb(connected);
      } catch (e) {
        console.error("[WS] connection callback error:", e);
      }
    });
  }

  private notifyDiagnosis(result: DiagnosisResult): void {
    this.diagnosisCallbacks.forEach((cb) => {
      try {
        cb(result);
      } catch (e) {
        console.error("[WS] diagnosis callback error:", e);
      }
    });
  }
}

// Export singleton instance
export const wsService = new WebSocketService();
