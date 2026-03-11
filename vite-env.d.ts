/// <reference types="vite/client" />

// File System Access API Types
interface FileSystemWritableFileStream extends WritableStream {
  write(data: string | BufferSource | Blob): Promise<void>;
  seek(position: number): Promise<void>;
  truncate(size: number): Promise<void>;
}

interface FileSystemFileHandle {
  createWritable(
    options?: Record<string, unknown>,
  ): Promise<FileSystemWritableFileStream>;
}

interface Window {
  showSaveFilePicker(
    options?: Record<string, unknown>,
  ): Promise<FileSystemFileHandle>;
}
