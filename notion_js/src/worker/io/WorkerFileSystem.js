const fs = require('fs/promises');
const path = require('path');

/**
 * @fileoverview I/O Abstraction Layer
 * @module worker/io/WorkerFileSystem
 * @description Provides a facade for Node.js file operations with safety checks
 * and explicit logging to prevent "ghost execution" failures.
 */

/**
 * @class WorkerFileSystem
 * @classdesc A facade for Node.js `fs` operations that enforces safety checks
 * and provides granular logging to prevent "ghost execution".
 * 
 * All write operations require absolute paths and log their actions explicitly,
 * ensuring visibility into every file I/O operation performed by workers.
 */
class WorkerFileSystem {
  /**
   * @constructor
   * @description Initializes the file system adapter.
   * @param {Logger} logger - Worker-context logger instance.
   */
  constructor(logger) {
    this.logger = logger;
    this.writtenFiles = new Set(); // Track files written in this session
  }
  
  /**
   * @method safeWrite
   * @async
   * @summary Writes content to a specific absolute path with safety checks.
   * @description Ensures the target directory exists (mkdir -p). Verifies the path
   * is absolute. Writes the file and logs the operation size and destination.
   * 
   * This method is the core "anti-ghost" mechanism that guarantees file writes
   * are visible and traceable.
   * 
   * @param {string} absolutePath - Target file path. MUST be absolute.
   * @param {string|Buffer} content - Content to write.
   * @returns {Promise<void>}
   * @throws {Error} If path is relative or write fails.
   * 
   * @example
   * await fs.safeWrite('/absolute/path/to/file.html', '<html>...</html>');
   * // Logs: [FS] Wrote 1024 bytes to /absolute/path/to/file.html
   */
  async safeWrite(absolutePath, content) {
    // Critical: Enforce absolute path requirement
    if (!path.isAbsolute(absolutePath)) {
      const error = new Error(
        `WorkerFileSystem.safeWrite() requires absolute path. Received: ${absolutePath}`
      );
      this.logger.error('FS', error.message);
      throw error;
    }
    
    try {
      // Ensure parent directory exists
      const directory = path.dirname(absolutePath);
      await this.ensureDir(directory);
      
      // Write the file
      await fs.writeFile(absolutePath, content, { encoding: 'utf-8' });
      
      // Calculate content size for logging
      const size = Buffer.isBuffer(content) 
        ? content.length 
        : Buffer.byteLength(content, 'utf-8');
      
      // Explicit logging: CRITICAL for debugging ghost writes
      this.logger.success('FS', `Wrote ${size} bytes to ${absolutePath}`);
      this.writtenFiles.add(absolutePath);
      
    } catch (error) {
      this.logger.error('FS', `Failed to write ${absolutePath}`, error);
      throw error;
    }
  }
  
  /**
   * @method ensureDir
   * @async
   * @summary Idempotent directory creation with recursive mkdir.
   * @description Creates the directory and all parent directories if they don't exist.
   * Does not throw if directory already exists.
   * 
   * @param {string} dirPath - Absolute directory path.
   * @returns {Promise<void>}
   * @throws {Error} If directory creation fails due to permissions or invalid path.
   */
  async ensureDir(dirPath) {
    try {
      await fs.mkdir(dirPath, { recursive: true });
      this.logger.debug('FS', `Ensured directory exists: ${dirPath}`);
    } catch (error) {
      // Ignore EEXIST errors (directory already exists)
      if (error.code !== 'EEXIST') {
        this.logger.error('FS', `Failed to create directory ${dirPath}`, error);
        throw error;
      }
    }
  }
  
  /**
   * @method getStats
   * @summary Returns statistics about file operations performed.
   * @returns {Object} Stats object with files written count.
   */
  getStats() {
    return {
      filesWritten: this.writtenFiles.size,
      writtenPaths: Array.from(this.writtenFiles)
    };
  }
}

module.exports = WorkerFileSystem;
