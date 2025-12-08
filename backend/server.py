"""
CAN to WebSocket Bridge Server
Reads CAN data from ZLG USB-CAN interface via ControlCAN.dll and forwards to frontend via WebSocket
"""

import asyncio
import json
import logging
import time
import os
import sys
import ctypes
from ctypes import Structure, c_uint, c_ubyte, c_ushort, POINTER, byref
from typing import Set, Dict, List, Optional

import websockets
from websockets.server import WebSocketServerProtocol

from config import (
    CANALYST_DEVICE_TYPE, CANALYST_DEVICE_INDEX, CANALYST_CHANNEL,
    CAN_BITRATE, WEBSOCKET_HOST, WEBSOCKET_PORT,
    CAN_TX_ID, BROADCAST_RATE
)
from can_protocol import (
    MachineState, MESSAGE_PARSERS, generate_control_packet
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Ctypes Definitions for ControlCAN.dll ---

class VCI_INIT_CONFIG(Structure):
    _fields_ = [("AccCode", c_uint),
                ("AccMask", c_uint),
                ("Reserved", c_uint),
                ("Filter", c_ubyte),
                ("Timing0", c_ubyte),
                ("Timing1", c_ubyte),
                ("Mode", c_ubyte)]

class VCI_CAN_OBJ(Structure):
    _fields_ = [("ID", c_uint),
                ("TimeStamp", c_uint),
                ("TimeFlag", c_ubyte),
                ("SendType", c_ubyte),
                ("RemoteFlag", c_ubyte),
                ("ExternFlag", c_ubyte),
                ("DataLen", c_ubyte),
                ("Data", c_ubyte * 8),
                ("Reserved", c_ubyte * 3)]

class VCI_CAN_OBJ_ARRAY(Structure):
    _fields_ = [('SIZE', c_ushort), ('STRUCT_ARRAY', POINTER(VCI_CAN_OBJ))]

    def __init__(self, num_of_structs):
        self.STRUCT_ARRAY = ctypes.cast((VCI_CAN_OBJ * num_of_structs)(), POINTER(VCI_CAN_OBJ))
        self.SIZE = num_of_structs
        self.ADDR = self.STRUCT_ARRAY[0]

# Baud Rate Setup (Timing0, Timing1)
BAUD_RATE_MAP = {
    10000:   (0x31, 0x1C),
    20000:   (0x18, 0x1C),
    40000:   (0x87, 0xFF),
    50000:   (0x09, 0x1C),
    80000:   (0x83, 0xFF),
    100000:  (0x04, 0x1C),
    125000:  (0x03, 0x1C),
    200000:  (0x81, 0xFA),
    250000:  (0x01, 0x1C),
    400000:  (0x80, 0xFA),
    500000:  (0x00, 0x1C),
    800000:  (0x00, 0x16),
    1000000: (0x00, 0x14),
}

import math
import random
from abc import ABC, abstractmethod

# ... (Configuration imports) ...
from config import (
    CANALYST_DEVICE_TYPE, CANALYST_DEVICE_INDEX, CANALYST_CHANNEL,
    CAN_BITRATE, WEBSOCKET_HOST, WEBSOCKET_PORT,
    CAN_TX_ID, BROADCAST_RATE, CAN_INTERFACE_TYPE
)
# ... (Protocol imports) ...

# ... (Ctypes Definitions and BAUD_RATE_MAP remain the same) ...

class CANDriver(ABC):
    """Abstract base class for CAN drivers"""
    
    @abstractmethod
    def open(self) -> bool:
        pass
        
    @abstractmethod
    def init_can(self, bitrate: int) -> bool:
        pass
        
    @abstractmethod
    def start(self) -> bool:
        pass
        
    @abstractmethod
    def close(self):
        pass
        
    @abstractmethod
    def receive(self, batch_size=100) -> List[Dict]:
        pass
        
    @abstractmethod
    def send(self, arbitration_id, data, is_extended=True) -> bool:
        pass


class VirtualDriver(CANDriver):
    """Virtual CAN driver for simulation and testing"""
    
    def __init__(self):
        self.running = False
        self.start_time = time.time()
        self.last_send_time = 0
        self.update_interval = 0.1 # 100ms
        logger.info("Initialized Virtual CAN Driver")
        
    def open(self) -> bool:
        logger.info("Virtual Device Opened")
        return True
        
    def init_can(self, bitrate: int) -> bool:
        logger.info(f"Virtual CAN initialized at {bitrate} bps")
        return True
        
    def start(self) -> bool:
        self.running = True
        logger.info("Virtual CAN Started")
        return True
        
    def close(self):
        self.running = False
        logger.info("Virtual CAN Closed")
        
    def receive(self, batch_size=100) -> List[Dict]:
        """Generate fake CAN messages"""
        if not self.running:
            return []
            
        current_time = time.time()
        if current_time - self.last_send_time < self.update_interval:
            return []
            
        self.last_send_time = current_time
        messages = []
        elapsed = current_time - self.start_time
        
        # 1. Heartbeat (0x18FF01F0)
        # Byte 0: Counter, Byte 1: Status=2(RUN)
        hb_counter = int(elapsed * 10) % 256
        data_status = bytes([hb_counter, 0x02, 0, 0, 0, 0, 0, 0])
        messages.append(self._create_msg(0x18FF01F0, data_status))
        
        # 2. Power Data (0x18FF02F0)
        # Sine wave simulation for voltage/current
        # Stack V: 150V + 50V*sin(t), Factor 0.01 -> 15000 + 5000*sin
        stack_v = int(15000 + 5000 * math.sin(elapsed * 0.5))
        stack_i = int(2000 + 1000 * math.sin(elapsed * 0.3)) # 200A +/- 100A
        dcf_v = 2400 # 24.0V
        dcf_i = int((stack_v * 0.01 * stack_i * 0.1 * 0.95) / (dcf_v * 0.01) * 10) # Calc based on power
        
        data_power = bytearray(8)
        data_power[0:2] = stack_v.to_bytes(2, 'little')
        data_power[2:4] = stack_i.to_bytes(2, 'little')
        data_power[4:6] = dcf_v.to_bytes(2, 'little')
        data_power[6:8] = dcf_i.to_bytes(2, 'little')
        messages.append(self._create_msg(0x18FF02F0, data_power))
        
        # 3. Sensors (0x18FF03F0)
        # Temp: 60C + 5 sin(t) -> (60+40)*10 = 1000
        temp = int((60 + 5 * math.sin(elapsed * 0.1) + 40) * 10)
        data_sensors = bytearray(8)
        data_sensors[0:2] = temp.to_bytes(2, 'little', signed=True) # Stack Temp
        data_sensors[2:4] = int((25+40)*10).to_bytes(2, 'little', signed=True) # Ambient
        data_sensors[4:6] = int(1200).to_bytes(2, 'little') # H2 Cyl Press 1200*0.01 = 12MPa
        messages.append(self._create_msg(0x18FF03F0, data_sensors))
        
        return messages
        
    def _create_msg(self, arbitration_id, data):
        return {
            'arbitration_id': arbitration_id,
            'data': bytearray(data),
            'is_extended_id': True,
            'timestamp': int(time.time() * 1000)
        }
        
    def send(self, arbitration_id, data, is_extended=True) -> bool:
        logger.info(f"[VirtualTX] ID: 0x{arbitration_id:08X} Data: {data.hex()}")
        return True


class ZLGDriver(CANDriver):
    """Wrapper for ZLG ControlCAN.dll"""
    
    def __init__(self, device_type=4, device_index=0, channel=0):
        self.device_type = device_type
        self.device_index = device_index
        self.channel = channel
        self.dll = None
        self.is_open = False
        self._load_dll()
        
    # ... (Keep existing _load_dll implementation) ...
    def _load_dll(self):
        """Load ControlCAN.dll from local directory"""
        if os.name == 'nt':
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                kerneldlls_dir = os.path.join(current_dir, 'kerneldlls')
                
                # Add paths to search order
                if os.path.exists(kerneldlls_dir):
                    os.add_dll_directory(kerneldlls_dir)
                os.add_dll_directory(current_dir)
                
                # Try loading
                dll_path = os.path.join(kerneldlls_dir, 'ControlCAN.dll')
                if not os.path.exists(dll_path):
                    dll_path = os.path.join(current_dir, 'ControlCAN.dll')
                    
                logger.info(f"Loading DLL from: {dll_path}")
                self.dll = ctypes.windll.LoadLibrary(dll_path)
                logger.info("ControlCAN.dll loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load ControlCAN.dll: {e}")
                raise
        else:
            logger.error("ZLGDriver only supports Windows")
            raise NotImplementedError("Windows only")

    def open(self) -> bool:
        if self.is_open:
            return True
        
        print(f"[DEBUG] Attempting to open ZLG device (Type={self.device_type}, Index={self.device_index})...", flush=True)
        logger.info(f"Attempting to open ZLG device (Type={self.device_type}, Index={self.device_index})")
        
        ret = self.dll.VCI_OpenDevice(self.device_type, self.device_index, 0)
        
        if ret == 1:
            print("[SUCCESS] VCI_OpenDevice succeeded!", flush=True)
            logger.info("VCI_OpenDevice succeeded")
            self.is_open = True
            return True
        else:
            print(f"[FAILED] VCI_OpenDevice failed. Return code: {ret}", flush=True)
            print("[INFO] Possible reasons:", flush=True)
            print("  1. USB-CAN adapter not connected", flush=True)
            print("  2. Driver not installed correctly", flush=True)
            print("  3. Device occupied by another program", flush=True)
            print("  4. Wrong device type in config.py", flush=True)
            logger.error(f"VCI_OpenDevice failed. Return code: {ret}")
            return False

    def init_can(self, bitrate: int) -> bool:
        if not self.is_open:
            print("[ERROR] Cannot init CAN: device not open", flush=True)
            return False
            
        if bitrate not in BAUD_RATE_MAP:
            print(f"[ERROR] Unsupported bitrate: {bitrate}", flush=True)
            logger.error(f"Unsupported bitrate: {bitrate}")
            return False
            
        t0, t1 = BAUD_RATE_MAP[bitrate]
        print(f"[DEBUG] Initializing CAN (Channel={self.channel}, Bitrate={bitrate}, T0=0x{t0:02X}, T1=0x{t1:02X})...", flush=True)
        
        # Config: AccCode=0x80000008, AccMask=0xFFFFFFFF (Accept All)
        config = VCI_INIT_CONFIG(0x80000008, 0xFFFFFFFF, 0, 0, t0, t1, 0)
        
        ret = self.dll.VCI_InitCAN(self.device_type, self.device_index, self.channel, byref(config))
        if ret == 1:
            print(f"[SUCCESS] VCI_InitCAN succeeded (Channel {self.channel}, Bitrate {bitrate})", flush=True)
            logger.info(f"VCI_InitCAN channel {self.channel} succeeded (Bitrate: {bitrate})")
            return True
        else:
            print(f"[FAILED] VCI_InitCAN failed. Return code: {ret}", flush=True)
            logger.error(f"VCI_InitCAN failed. Return code: {ret}")
            return False

    def start(self) -> bool:
        if not self.is_open:
            print("[ERROR] Cannot start CAN: device not open", flush=True)
            return False
        
        print(f"[DEBUG] Starting CAN (Channel={self.channel})...", flush=True)
        ret = self.dll.VCI_StartCAN(self.device_type, self.device_index, self.channel)
        
        if ret == 1:
            print("[SUCCESS] VCI_StartCAN succeeded! CAN bus is ACTIVE.", flush=True)
            logger.info("VCI_StartCAN succeeded")
            return True
        else:
            print(f"[FAILED] VCI_StartCAN failed. Return code: {ret}", flush=True)
            logger.error(f"VCI_StartCAN failed. Return code: {ret}")
            return False

    def close(self):
        if self.is_open:
            self.dll.VCI_CloseDevice(self.device_type, self.device_index)
            self.is_open = False
            logger.info("Device closed")

    def receive(self, batch_size=100) -> List[Dict]:
        if not self.is_open:
            return []
            
        rx_buffer = VCI_CAN_OBJ_ARRAY(batch_size)
        num_frames = self.dll.VCI_Receive(self.device_type, self.device_index, self.channel, 
                                        byref(rx_buffer.ADDR), batch_size, 0)
        
        messages = []
        if num_frames > 0:
            for i in range(num_frames):
                obj = rx_buffer.STRUCT_ARRAY[i]
                is_extended = (obj.ExternFlag == 1)
                messages.append({
                    'arbitration_id': obj.ID,
                    'data': bytearray(obj.Data)[:obj.DataLen],
                    'is_extended_id': is_extended,
                    'timestamp': obj.TimeStamp
                })
        return messages

    def send(self, arbitration_id, data, is_extended=True) -> bool:
        if not self.is_open:
            return False
            
        data_len = len(data)
        ubyte_array_8 = c_ubyte * 8
        ubyte_array_3 = c_ubyte * 3
        
        data_buf = ubyte_array_8(*list(data) + [0] * (8 - data_len))
        reserved = ubyte_array_3(0, 0, 0)
        
        vci_obj = VCI_CAN_OBJ(
            arbitration_id, 0, 0, 0, 0, 
            1 if is_extended else 0, 
            data_len, data_buf, reserved
        )
        
        ret = self.dll.VCI_Transmit(self.device_type, self.device_index, self.channel, byref(vci_obj), 1)
        return ret == 1



class SimpleMessage:
    def __init__(self, arbitration_id, data, is_extended_id=True):
        self.arbitration_id = arbitration_id
        self.data = data
        self.is_extended_id = is_extended_id

class CANWebSocketServer:
    """CAN to WebSocket bridge server"""
    
    def __init__(self):
        self.driver: CANDriver = None
        self.machine_state = MachineState()
        self.clients: Set[WebSocketServerProtocol] = set()
        self.running = False
        
    async def start(self):
        """Start the server"""
        self.running = True
        
        # Factory: Initialize Driver based on config
        logger.info(f"Initializing Driver Type: {CAN_INTERFACE_TYPE}")
        
        try:
            if CAN_INTERFACE_TYPE == "virtual":
                self.driver = VirtualDriver()
            elif CAN_INTERFACE_TYPE == "zlg":
                self.driver = ZLGDriver(CANALYST_DEVICE_TYPE, CANALYST_DEVICE_INDEX, CANALYST_CHANNEL)
            else:
                logger.error(f"Unknown interface type: {CAN_INTERFACE_TYPE}. Defaulting to ZLG.")
                self.driver = ZLGDriver(CANALYST_DEVICE_TYPE, CANALYST_DEVICE_INDEX, CANALYST_CHANNEL)
            
            if not self.driver.open():
                logger.error("Failed to open CAN device. Exiting.")
                return

            if not self.driver.init_can(CAN_BITRATE):
                logger.error(f"Failed to initialize CAN channel with bitrate {CAN_BITRATE}. Exiting.")
                self.driver.close()
                return
                
            if not self.driver.start():
                logger.error("Failed to start CAN communication. Exiting.")
                self.driver.close()
                return
                
            logger.info(f"✓ CAN Bus initialized ({CAN_INTERFACE_TYPE} Driver)")
            
        except Exception as e:
            logger.error(f"Exception during driver initialization: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # ... (Rest of start method remains the same) ...

        
        # Start CAN receive task
        can_task = asyncio.create_task(self.can_receive_loop())
        
        # Start WebSocket broadcast task
        broadcast_task = asyncio.create_task(self.broadcast_loop())
        
        # Start WebSocket server
        logger.info(f"Starting WebSocket server on {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        try:
            async with websockets.serve(self.handle_client, WEBSOCKET_HOST, WEBSOCKET_PORT):
                logger.info("✓ WebSocket server started")
                logger.info(f"Frontend should connect to: ws://localhost:{WEBSOCKET_PORT}")
                
                # Keep running
                await asyncio.gather(can_task, broadcast_task)
        except Exception as e:
            logger.error(f"WebSocket server failed: {e}")
            
    async def can_receive_loop(self):
        """Continuously read CAN messages"""
        logger.info("CAN receive loop started")
        
        while self.running:
            try:
                # Poll for messages
                # Since VCI_Receive is blocking/polling in C, we should be careful not to block event loop too long.
                # However, with wait_time=0 it should be non-blocking.
                # We can run it in executor if it blocks, but let's try direct call first.
                
                messages = self.driver.receive(batch_size=50) # Batch read
                
                if messages:
                    for msg_dict in messages:
                        # Convert to object for compatibility
                        msg = SimpleMessage(msg_dict['arbitration_id'], msg_dict['data'], msg_dict['is_extended_id'])
                        await self.process_can_message(msg)
                
                # Sleep briefly to yield to event loop
                await asyncio.sleep(0.01) 
                    
            except Exception as e:
                logger.error(f"Error in CAN receive loop: {e}")
                await asyncio.sleep(0.1)
                
    async def process_can_message(self, msg):
        """Process incoming CAN message"""
        can_id = msg.arbitration_id
        
        # Check if this is a known message
        if can_id in MESSAGE_PARSERS:
            try:
                parser = MESSAGE_PARSERS[can_id]
                parser(msg.data, self.machine_state)
                
                # Update timestamp
                self.machine_state.last_update = int(time.time() * 1000)
                self.machine_state.connected = True
                
                # Log occasionally (every 100 messages for ID 0x18FF01F0)
                if can_id == 0x18FF01F0 and self.machine_state.status.get("heartbeat", 0) % 100 == 0:
                    logger.info(f"CAN RX: 0x{can_id:08X} - Heartbeat: {self.machine_state.status.get('heartbeat')}")
            except Exception as e:
                logger.warning(f"Error parsing message 0x{can_id:08X}: {e}")
    
    async def broadcast_loop(self):
        """Broadcast machine state to all connected clients periodically"""
        interval = 1.0 / BROADCAST_RATE
        
        while self.running:
            if self.clients:
                try:
                    message = json.dumps({
                        "type": "machine_state",
                        "data": self.machine_state.to_dict()
                    })
                    
                    # Send to all connected clients
                    disconnected = set()
                    for client in self.clients:
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected.add(client)
                    
                    # Remove disconnected clients
                    self.clients -= disconnected
                except Exception as e:
                    logger.error(f"Error broadcasting state: {e}")
            
            await asyncio.sleep(interval)
    
    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a WebSocket client connection"""
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")
        
        # Add to clients set
        self.clients.add(websocket)
        
        try:
            # Send initial state
            initial_message = json.dumps({
                "type": "machine_state",
                "data": self.machine_state.to_dict()
            })
            await websocket.send(initial_message)
            
            # Listen for control commands
            async for message in websocket:
                await self.handle_client_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")
        except Exception as e:
            logger.error(f"Error handling client {client_addr}: {e}")
        finally:
            # Remove from clients set
            self.clients.discard(websocket)
    
    async def handle_client_message(self, message: str):
        """Handle incoming WebSocket message from client"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "control":
                # Extract control data
                control = data.get("data", {})
                
                # Generate CAN packet
                can_data = generate_control_packet(control)
                
                # Send to CAN bus
                success = self.driver.send(CAN_TX_ID, can_data, is_extended=True)
                
                if success:
                    logger.info(f"CAN TX: 0x{CAN_TX_ID:08X} - Command: {control.get('command', 0)}")
                else:
                    logger.warning(f"Failed to send CAN TX: 0x{CAN_TX_ID:08X}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.driver:
            self.driver.close()
            logger.info("CAN bus closed")


async def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("H2 FCU CAN to WebSocket Bridge Server (Direct DLL)")
    logger.info("=" * 60)
    
    server = CANWebSocketServer()
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        server.stop()


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("启动 H2 FCU Backend Server...", flush=True)
    print("=" * 60, flush=True)
    asyncio.run(main())
