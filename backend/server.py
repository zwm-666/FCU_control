"""
CAN to WebSocket Bridge Server
Reads CAN data from ZLG USB-CAN interface and forwards to frontend via WebSocket
"""

import asyncio
import json
import logging
import time
from typing import Set
import can
import websockets
from websockets.server import WebSocketServerProtocol

from config import (
    CAN_INTERFACE, CAN_BITRATE,
    WEBSOCKET_HOST, WEBSOCKET_PORT,
    CAN_TX_ID, BROADCAST_RATE
)
import config  # Import module for dynamic attribute access
from can_protocol import (
    MachineState, MESSAGE_PARSERS, generate_control_packet
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CANWebSocketServer:
    """CAN to WebSocket bridge server"""
    
    def __init__(self):
        self.can_bus: can.Bus = None
        self.machine_state = MachineState()
        self.clients: Set[WebSocketServerProtocol] = set()
        self.running = False
        
    async def start(self):
        """Start the server"""
        self.running = True
        
        # Initialize CAN bus
        try:
            logger.info(f"Initializing CAN interface: {CAN_INTERFACE}")
            
            if CAN_INTERFACE == "zlgcan":
                from config import ZLG_DEVICE_TYPE, ZLG_DEVICE_INDEX, ZLG_CHANNEL
                logger.info(f"ZLG Device Type: {ZLG_DEVICE_TYPE}, Index: {ZLG_DEVICE_INDEX}, Channel: {ZLG_CHANNEL}")
                logger.info(f"Bitrate: {CAN_BITRATE}")
                
                self.can_bus = can.Bus(
                    interface=CAN_INTERFACE,
                    channel=ZLG_CHANNEL,
                    device=ZLG_DEVICE_INDEX,
                    device_type=ZLG_DEVICE_TYPE,
                    bitrate=CAN_BITRATE
                )
            else:
                # For other interfaces (socketcan, virtual, etc.)
                logger.info(f"Channel: {getattr(config, 'CAN_CHANNEL', 'N/A')}, Bitrate: {CAN_BITRATE}")
                self.can_bus = can.Bus(
                    interface=CAN_INTERFACE,
                    channel=getattr(config, 'CAN_CHANNEL', 0),
                    bitrate=CAN_BITRATE
                )
            
            logger.info("✓ CAN bus initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize CAN bus: {e}")
            logger.error("Please check:")
            logger.error("  - ControlCAN.dll is in system path or same directory")
            logger.error("  - ZLG USB-CAN device is connected")
            logger.error("  - Device type and channel are correct in config.py")
            logger.error("  - No other programs are using the device")
            return
        
        # Start CAN receive task
        can_task = asyncio.create_task(self.can_receive_loop())
        
        # Start WebSocket broadcast task
        broadcast_task = asyncio.create_task(self.broadcast_loop())
        
        # Start WebSocket server
        logger.info(f"Starting WebSocket server on {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        async with websockets.serve(self.handle_client, WEBSOCKET_HOST, WEBSOCKET_PORT):
            logger.info("✓ WebSocket server started")
            logger.info(f"Frontend should connect to: ws://localhost:{WEBSOCKET_PORT}")
            
            # Keep running
            await asyncio.gather(can_task, broadcast_task)
    
    async def can_receive_loop(self):
        """Continuously read CAN messages"""
        logger.info("CAN receive loop started")
        
        while self.running:
            try:
                # Read CAN message with timeout
                msg = self.can_bus.recv(timeout=0.1)
                
                if msg is not None:
                    await self.process_can_message(msg)
                    
            except Exception as e:
                logger.error(f"Error in CAN receive loop: {e}")
                await asyncio.sleep(0.1)
    
    async def process_can_message(self, msg: can.Message):
        """Process incoming CAN message"""
        can_id = msg.arbitration_id
        
        # Check if this is a known message
        if can_id in MESSAGE_PARSERS:
            parser = MESSAGE_PARSERS[can_id]
            parser(msg.data, self.machine_state)
            
            # Update timestamp
            self.machine_state.last_update = int(time.time() * 1000)
            self.machine_state.connected = True
            
            # Log occasionally (every 100 messages for ID 0x18FF01F0)
            if can_id == 0x18FF01F0 and self.machine_state.status["heartbeat"] % 100 == 0:
                logger.debug(f"CAN RX: 0x{can_id:08X} - Heartbeat: {self.machine_state.status['heartbeat']}")
    
    async def broadcast_loop(self):
        """Broadcast machine state to all connected clients periodically"""
        interval = 1.0 / BROADCAST_RATE
        
        while self.running:
            if self.clients:
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
                can_msg = can.Message(
                    arbitration_id=CAN_TX_ID,
                    data=can_data,
                    is_extended_id=True
                )
                
                self.can_bus.send(can_msg)
                
                logger.info(f"CAN TX: 0x{CAN_TX_ID:08X} - Command: {control.get('command', 0)}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.can_bus:
            self.can_bus.shutdown()
            logger.info("CAN bus closed")


async def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("H2 FCU CAN to WebSocket Bridge Server")
    logger.info("=" * 60)
    
    server = CANWebSocketServer()
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        server.stop()


if __name__ == "__main__":
    asyncio.run(main())
