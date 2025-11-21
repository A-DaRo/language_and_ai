/**
 * @fileoverview Centralized Logging System with Strategy Pattern
 * @module core/Logger
 */

const ConsoleStrategy = require('./logger/ConsoleStrategy');
const FileStrategy = require('./logger/FileStrategy');

/**
 * @class Logger
 * @description Singleton logger that dispatches messages to multiple strategies.
 * Supports dynamic strategy switching for flexible output routing.
 */
class Logger {
  /**
   * @private
   * @static
   * @type {Logger}
   * @description Singleton instance
   */
  static _instance = null;

  /**
   * @private
   * @description Private constructor to enforce singleton pattern
   */
  constructor() {
    if (Logger._instance) {
      throw new Error('Logger is a singleton. Use getInstance() instead.');
    }
    
    this.strategies = [];
    this.startTime = Date.now();
    this._initialized = false;
  }

  /**
   * Get the singleton instance of the Logger
   * @static
   * @returns {Logger} The singleton instance
   * @example
   * const logger = Logger.getInstance();
   */
  static getInstance() {
    if (!Logger._instance) {
      Logger._instance = new Logger();
    }
    return Logger._instance;
  }

  /**
   * Initialize logger with strategies
   * @param {Object} options - Configuration options
   * @param {boolean} [options.console=false] - Enable console strategy
   * @param {boolean} [options.file=false] - Enable file strategy
   * @param {string} [options.outputDir='./output'] - Output directory for file logs
   * @param {string} [options.logSubdir='logs'] - Subdirectory for log files
   */
  init(options = {}) {
    if (this._initialized) {
      return; // Prevent re-initialization
    }

    if (options.console) {
      this.addStrategy(new ConsoleStrategy());
    }

    if (options.file) {
      const outputDir = options.outputDir || './output';
      const logSubdir = options.logSubdir !== undefined ? options.logSubdir : 'logs';
      this.addStrategy(new FileStrategy(outputDir, { subdir: logSubdir }));
    }

    this._initialized = true;
  }

  /**
   * Add a logging strategy
   * @param {LogStrategy} strategy - Strategy instance to add
   */
  addStrategy(strategy) {
    this.strategies.push(strategy);
  }

  /**
   * Remove a specific strategy
   * @param {LogStrategy} strategy - Strategy instance to remove
   */
  removeStrategy(strategy) {
    const index = this.strategies.indexOf(strategy);
    if (index !== -1) {
      this.strategies.splice(index, 1);
    }
  }

  /**
   * Switch between console and dashboard modes
   * @param {string} mode - 'console' or 'dashboard'
   * @param {Object} [context={}] - Context object with dashboardInstance if mode is 'dashboard'
   * @description Safely switches UI strategies without affecting FileStrategy
   */
  switchMode(mode, context = {}) {
    // Remove existing UI strategies (Console or Dashboard)
    this.strategies = this.strategies.filter(s => {
      const isConsole = s.constructor.name === 'ConsoleStrategy';
      const isDashboard = s.constructor.name === 'DashboardStrategy';
      return !(isConsole || isDashboard);
    });

    // Add new UI strategy
    if (mode === 'dashboard' && context.dashboardInstance) {
      const DashboardStrategy = require('./logger/DashboardStrategy');
      this.addStrategy(new DashboardStrategy(context.dashboardInstance));
    } else if (mode === 'console') {
      this.addStrategy(new ConsoleStrategy());
    }
  }

  /**
   * Get current active UI strategy name (for debugging)
   * @returns {string|null} The name of the current UI strategy or null if none
   */
  getCurrentUIStrategy() {
    const uiStrategy = this.strategies.find(s => {
      const name = s.constructor.name;
      return name === 'ConsoleStrategy' || name === 'DashboardStrategy';
    });
    return uiStrategy ? uiStrategy.constructor.name : null;
  }

  /**
   * Log an informational message
   * @param {string} category - Message category
   * @param {string} message - Log message
   */
  info(category, message) {
    this._dispatch('info', category, message);
  }

  /**
   * Log a success message
   * @param {string} category - Message category
   * @param {string} message - Success message
   */
  success(category, message) {
    this._dispatch('success', category, message);
  }

  /**
   * Log a warning message
   * @param {string} category - Message category
   * @param {string} message - Warning message
   */
  warn(category, message) {
    this._dispatch('warn', category, message);
  }

  /**
   * Log an error message
   * @param {string} category - Message category
   * @param {string} message - Error message
   * @param {Error} [error] - Optional error object
   */
  error(category, message, error) {
    const meta = error ? {
      message: error.message,
      stack: error.stack,
      name: error.name
    } : null;
    this._dispatch('error', category, message, meta);
  }

  /**
   * Log a debug message
   * @param {string} category - Message category
   * @param {string} message - Debug message
   * @param {Object} [meta] - Optional metadata
   */
  debug(category, message, meta) {
    this._dispatch('debug', category, message, meta);
  }

  /**
   * Print a visual separator
   * @param {string} [text=''] - Optional text to display in separator
   */
  separator(text = '') {
    const line = text ? `═══════════════ ${text} ═══════════════` : '═'.repeat(60);
    this.info('SYSTEM', line);
  }

  /**
   * Get elapsed time since logger initialization
   * @returns {number} Elapsed time in milliseconds
   */
  getElapsedTime() {
    return Date.now() - this.startTime;
  }

  /**
   * Close all strategies and cleanup
   */
  close() {
    this.strategies.forEach(strategy => {
      if (strategy.close) {
        strategy.close();
      }
    });
    this.strategies = [];
  }

  /**
   * @private
   * @param {string} level - Log level
   * @param {string} category - Message category
   * @param {string} message - Log message
   * @param {Object} [meta] - Optional metadata
   */
  _dispatch(level, category, message, meta) {
    this.strategies.forEach(strategy => {
      try {
        strategy.log(level, category, message, meta);
      } catch (err) {
        // Prevent strategy failures from crashing the application
        console.error(`[Logger] Strategy error: ${err.message}`);
      }
    });
  }
}

module.exports = Logger;
