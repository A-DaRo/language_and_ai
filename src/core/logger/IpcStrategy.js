/**
 * @fileoverview IPC Logging Strategy
 * @module core/logger/IpcStrategy
 */

const LogStrategy = require('./LogStrategy');

/**
 * @class IpcStrategy
 * @extends LogStrategy
 * @description Sends log entries to the Master process via Node.js IPC.
 * Used by Workers to prevent stdout pollution in the terminal.
 */
class IpcStrategy extends LogStrategy {
  constructor() {
    super();
  }

  log(level, category, message, meta) {
    // specific check to avoid circular JSON errors or massive payloads
    const safeMeta = meta ? this._sanitizeMeta(meta) : null;

    if (process.send) {
      process.send({
        type: 'IPC_LOG',
        payload: {
          level,
          category,
          message,
          meta: safeMeta,
          timestamp: Date.now()
        }
      });
    }
  }

  /**
   * Sanitize metadata to remove circular references and convert Errors
   * @private
   * @param {*} meta - Metadata to sanitize
   * @returns {*} Sanitized metadata
   */
  _sanitizeMeta(meta) {
    const seen = new WeakSet();
    return JSON.parse(JSON.stringify(meta, (key, value) => {
      // Handle Error objects
      if (value instanceof Error) {
        return {
          message: value.message,
          stack: value.stack,
          name: value.name
        };
      }
      
      // Handle circular references
      if (typeof value === 'object' && value !== null) {
        if (seen.has(value)) {
          return '[Circular]';
        }
        seen.add(value);
      }
      
      return value;
    }));
  }

  // Helper to handle circular references in errors (kept for backwards compatibility)
  _replacer(key, value) {
    if (value instanceof Error) {
      return {
        message: value.message,
        stack: value.stack,
        name: value.name
      };
    }
    return value;
  }
}

module.exports = IpcStrategy;
