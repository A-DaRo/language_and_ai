/**
 * @fileoverview Base Strategy Interface for Logging
 * @module core/logger/LogStrategy
 */

/**
 * @class LogStrategy
 * @abstract
 * @description Abstract base class for all logging strategies.
 * Implementations define where and how log messages are output.
 */
class LogStrategy {
  /**
   * @method log
   * @abstract
   * @param {string} level - Log level: 'info', 'warn', 'error', 'debug', 'success'
   * @param {string} category - Source module/category name
   * @param {string} message - Log message
   * @param {Object} [meta] - Optional metadata object
   * @description Must be implemented by all strategy subclasses
   */
  log(level, category, message, meta) {
    throw new Error('LogStrategy.log() must be implemented by subclass');
  }

  /**
   * @method close
   * @description Optional cleanup method. Subclasses can override if needed.
   */
  close() {
    // Default: no-op
  }
}

module.exports = LogStrategy;
