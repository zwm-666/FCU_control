/**
 * WebSocket Service for CAN Data Communication
 * Connects to Python backend WebSocket server
 */

import { MachineState, ControlState, DiagnosisResult, DiagnosisLabel } from '../types';

type MachineStateCallback = (state: MachineState) => void;
type ConnectionCallback = (connected: boolean) => void;
type DiagnosisCallback = (result: DiagnosisResult) => void;

const isDev = import.meta.env.DEV;

class WebSocketService {
    private ws: WebSocket | null = null;
    private reconnectTimer: number | null = null;
    private machineStateCallbacks: Set<MachineStateCallback> = new Set();
    private connectionCallbacks: Set<ConnectionCallback> = new Set();
    private diagnosisCallbacks: Set<DiagnosisCallback> = new Set();
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 10;
    private reconnectDelay = 2000;

    connect(url: string = 'ws://localhost:8765'): void {
        if (this.ws?.readyState === WebSocket.OPEN) {
            return;
        }

        if (isDev) console.log(`Connecting to WebSocket: ${url}`);

        try {
            this.ws = new WebSocket(url);

            this.ws.onopen = () => {
                if (isDev) console.log('WebSocket connected');
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
                if (isDev) console.log('WebSocket disconnected');
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
        if (isDev) console.log(`Reconnecting (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

        this.reconnectTimer = window.setTimeout(() => {
            this.connect(url);
        }, this.reconnectDelay);
    }

    private handleMessage(data: string): void {
        try {
            const message = JSON.parse(data);

            if (message.type === 'machine_state') {
                const state = message.data as MachineState;
                this.notifyMachineState(state);

                if (message.diagnosis) {
                    this.notifyDiagnosis(message.diagnosis as DiagnosisResult);
                }
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
            if (isDev) console.log('TX Control:', control);
        }
    }

    sendDiagnosisFeedback(label: DiagnosisLabel): void {
        if (this.ws?.readyState === WebSocket.OPEN) {
            const message = {
                type: 'diagnosis_feedback',
                data: { label }
            };

            this.ws.send(JSON.stringify(message));
            if (isDev) console.log('TX Diagnosis Feedback:', label);
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
