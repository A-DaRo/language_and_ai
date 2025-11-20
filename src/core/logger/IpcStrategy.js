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
    const safeMeta = meta ? JSON.parse(JSON.stringify(meta, this._replacer)) : null;

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

  // Helper to handle circular references in errors
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
