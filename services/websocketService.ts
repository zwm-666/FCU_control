/**
 * WebSocket Service for CAN Data Communication
 * Connects to Python backend WebSocket server
 */

import { MachineState, ControlState, DiagnosisResult, DiagnosisLabel } from '../types';

type MachineStateCallback = (state: MachineState) => void;
type ConnectionCallback = (connected: boolean) => void;
type DiagnosisCallback = (result: DiagnosisResult) => void;

class WebSocketService {
    private ws: WebSocket | null = null;
    private reconnectTimer: number | null = null;
    private machineStateCallbacks: Set<MachineStateCallback> = new Set();
    private connectionCallbacks: Set<ConnectionCallback> = new Set();
    private diagnosisCallbacks: Set<DiagnosisCallback> = new Set();
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 10;
    private reconnectDelay = 2000; // 2 seconds

    connect(url: string = 'ws://localhost:8765'): void {
        if (this.ws?.readyState === WebSocket.OPEN) {
            console.log('WebSocket already connected');
            return;
        }

        console.log(`Connecting to WebSocket: ${url}`);

        try {
            this.ws = new WebSocket(url);

            this.ws.onopen = () => {
                console.log('✓ WebSocket connected');
                this.reconnectAttempts = 0;
                this.notifyConnection(true);
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(event.data);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.notifyConnection(false);
                this.attemptReconnect(url);
            };

        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.notifyConnection(false);
        }
    }

    disconnect(): void {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.reconnectAttempts = 0;
        this.notifyConnection(false);
    }

    private attemptReconnect(url: string): void {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            return;
        }

        this.reconnectAttempts++;
        console.log(`Reconnecting in ${this.reconnectDelay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

        this.reconnectTimer = window.setTimeout(() => {
            this.connect(url);
        }, this.reconnectDelay);
    }

    private handleMessage(data: string): void {
        try {
            const message = JSON.parse(data);

            if (message.type === 'machine_state') {
                const state = message.data as MachineState;
                console.log('📥 收到状态更新:', {
                    connected: state.connected,
                    lastUpdate: state.lastUpdate,
                    stackVoltage: state.power?.stackVoltage,
                    systemState: state.status?.state
                });
                this.notifyMachineState(state);

                // 处理诊断结果
                if (message.diagnosis) {
                    this.notifyDiagnosis(message.diagnosis as DiagnosisResult);
                }
            } else {
                console.log('📥 收到未知类型消息:', message.type);
            }

        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }

    sendControl(control: ControlState): void {
        if (this.ws?.readyState === WebSocket.OPEN) {
            const message = {
                type: 'control',
                data: control
            };

            this.ws.send(JSON.stringify(message));
            console.log('TX Control:', control);
        } else {
            console.warn('Cannot send control: WebSocket not connected');
        }
    }

    sendDiagnosisFeedback(label: DiagnosisLabel): void {
        if (this.ws?.readyState === WebSocket.OPEN) {
            const message = {
                type: 'diagnosis_feedback',
                data: { label }
            };

            this.ws.send(JSON.stringify(message));
            console.log('TX Diagnosis Feedback:', label);
        } else {
            console.warn('Cannot send diagnosis feedback: WebSocket not connected');
        }
    }

    onMachineState(callback: MachineStateCallback): () => void {
        this.machineStateCallbacks.add(callback);

        // Return unsubscribe function
        return () => {
            this.machineStateCallbacks.delete(callback);
        };
    }

    onConnection(callback: ConnectionCallback): () => void {
        this.connectionCallbacks.add(callback);

        // Call immediately with current state
        callback(this.ws?.readyState === WebSocket.OPEN);

        // Return unsubscribe function
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

    private notifyMachineState(state: MachineState): void {
        this.machineStateCallbacks.forEach(callback => {
            try {
                callback(state);
            } catch (error) {
                console.error('Error in machine state callback:', error);
            }
        });
    }

    private notifyConnection(connected: boolean): void {
        this.connectionCallbacks.forEach(callback => {
            try {
                callback(connected);
            } catch (error) {
                console.error('Error in connection callback:', error);
            }
        });
    }

    private notifyDiagnosis(result: DiagnosisResult): void {
        this.diagnosisCallbacks.forEach(callback => {
            try {
                callback(result);
            } catch (error) {
                console.error('Error in diagnosis callback:', error);
            }
        });
    }

    isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }
}

// Export singleton instance
export const wsService = new WebSocketService();
