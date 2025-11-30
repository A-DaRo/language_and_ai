/**
 * @fileoverview Console Logging Strategy
 * @module core/logger/ConsoleStrategy
 */

const LogStrategy = require('./LogStrategy');

/**
 * @class ConsoleStrategy
 * @extends LogStrategy
 * @description Logs messages to the console with color formatting
 */
class ConsoleStrategy extends LogStrategy {
  /**
   * Log a message to the console
   * @param {string} level - Log level (info, success, warn, error, debug)
   * @param {string} category - Message category
   * @param {string} message - Log message
   * @param {Object} [meta] - Optional metadata
   */
  log(level, category, message, meta) {
    const timestamp = new Date().toISOString();
    const prefix = `[${timestamp}] [${level.toUpperCase()}] [${category}]`;
    
    let output;
    switch (level) {
      case 'error':
        output = `\x1b[31m${prefix}\x1b[0m ${message}`;
        console.error(output);
        if (meta) {
          console.error('  ', JSON.stringify(meta, null, 2));
        }
        break;
      
      case 'warn':
        output = `\x1b[33m${prefix}\x1b[0m ${message}`;
        console.warn(output);
        if (meta) {
          console.warn('  ', JSON.stringify(meta, null, 2));
        }
        break;
      
      case 'success':
        output = `\x1b[32m${prefix}\x1b[0m ${message}`;
        console.log(output);
        break;
      
      case 'debug':
        output = `\x1b[90m${prefix}\x1b[0m ${message}`;
        console.log(output);
        if (meta) {
          console.log('  ', JSON.stringify(meta, null, 2));
        }
        break;
      
      case 'info':
      default:
        output = `${prefix} ${message}`;
        console.log(output);
        break;
    }
  }
}

module.exports = ConsoleStrategy;
