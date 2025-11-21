/**
 * @fileoverview Dashboard Logging Strategy
 * @module core/logger/DashboardStrategy
 */

const LogStrategy = require('./LogStrategy');

/**
 * @class DashboardStrategy
 * @extends LogStrategy
 * @description Sends log messages to the TerminalDashboard footer for display.
 * Maintains a circular buffer of recent log entries to prevent UI clutter.
 */
class DashboardStrategy extends LogStrategy {
  /**
   * @constructor
   * @param {TerminalDashboard} dashboardInstance - The dashboard to send logs to
   * @param {Object} [options={}] - Configuration options
   * @param {number} [options.bufferSize=5] - Number of recent logs to maintain
   */
  constructor(dashboardInstance, options = {}) {
    super();
    this.dashboard = dashboardInstance;
    this.bufferSize = options.bufferSize || 5;
    this.recentLogs = [];
  }

  /**
   * @method log
   * @param {string} level - Log level
   * @param {string} category - Message category
   * @param {string} message - Log message
   * @param {Object} [meta] - Optional metadata (ignored for dashboard)
   */
  log(level, category, message, meta) {
    // Skip debug messages to reduce UI clutter
    if (level === 'debug') {
      return;
    }

    // Filter out navigation logs from the footer to prevent banner-like artifacts
    if (message && message.includes('Navigating to:')) {
      return;
    }

    // Format the log entry
    const time = new Date().toLocaleTimeString();
    const icon = this._getLevelIcon(level);
    const formattedLog = `${time} ${icon} [${category}] ${message}`;

    // Add to circular buffer
    this.recentLogs.push(formattedLog);
    if (this.recentLogs.length > this.bufferSize) {
      this.recentLogs.shift();
    }

    // Update dashboard footer with most recent log
    this.dashboard.updateFooter(formattedLog);
  }

  /**
   * @private
   * @method _getLevelIcon
   * @param {string} level - Log level
   * @returns {string} Icon for the level
   */
  _getLevelIcon(level) {
    switch (level) {
      case 'error':
        return '[ERROR]';
      case 'warn':
        return '[WARN]';
      case 'success':
        return '[OK]';
      case 'info':
      default:
        return '[INFO]';
    }
  }
}

module.exports = DashboardStrategy;
