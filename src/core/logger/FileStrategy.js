/**
 * @fileoverview File Logging Strategy
 * @module core/logger/FileStrategy
 */

const fs = require('fs');
const path = require('path');
const LogStrategy = require('./LogStrategy');

/**
 * @class FileStrategy
 * @extends LogStrategy
 * @description Writes log messages to a markdown-formatted file.
 * Creates a timestamped log file in the specified directory.
 */
class FileStrategy extends LogStrategy {
  /**
   * @constructor
   * @param {string} baseDir - Base output directory (e.g., './output')
   * @param {Object} [options={}] - Configuration options
   * @param {string} [options.subdir='logs'] - Subdirectory for log files
   */
  constructor(baseDir, options = {}) {
    super();
    
    const subdir = options.subdir !== undefined ? options.subdir : 'logs';
    const logsDir = (subdir !== '' && subdir !== null) ? path.join(baseDir, subdir) : baseDir;
    
    // Create logs directory if it doesn't exist
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }

    // Generate timestamped filename
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('.')[0];
    this.filepath = path.join(logsDir, `run-${timestamp}.md`);
    
    // Open write stream
    this.stream = fs.createWriteStream(this.filepath, { flags: 'a' });
  }

  /**
   * @method log
   * @param {string} level - 'info', 'warn', 'error', 'debug', 'success'
   * @param {string} category - Source module name
   * @param {string} message - Log message
   * @param {Object} [meta] - Optional metadata
   */
  log(level, category, message, meta) {
    const time = new Date().toLocaleTimeString();
    const levelTag = `[${level.toUpperCase()}]`;
    const categoryTag = `[${category}]`;
    let line = `${time} ${levelTag} ${categoryTag} ${message}\n`;

    if (meta && (level === 'error' || level === 'debug')) {
      line += `  └─ ${JSON.stringify(meta, null, 2).replace(/\n/g, '\n     ')}\n`;
    }
    
    this.stream.write(line);
  }

  /**
   * @method close
   * @description Closes the file stream and writes footer
   */
  close() {
    if (this.stream && !this.stream.destroyed) {
      this.stream.end();
    }
  }

  /**
   * @method getFilePath
   * @returns {string} Full path to the log file
   */
  getFilePath() {
    return this.filepath;
  }
}

module.exports = FileStrategy;
