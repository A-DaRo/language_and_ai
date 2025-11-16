/**
 * Logger utility for structured logging
 */
class Logger {
  constructor() {
    this.startTime = Date.now();
  }
  
  /**
   * Get elapsed time since logger creation
   */
  getElapsedTime() {
    const elapsed = Date.now() - this.startTime;
    const seconds = Math.floor(elapsed / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  }
  
  /**
   * Log with category prefix
   */
  log(category, message) {
    const timestamp = new Date().toLocaleTimeString();
    console.log(`[${timestamp}] [${category}] ${message}`);
  }
  
  info(category, message) {
    this.log(category, message);
  }
  
  success(category, message) {
    this.log(category, `SUCCESS: ${message}`);
  }
  
  error(category, message, error = null) {
    const errorMsg = error ? `${message}: ${error.message}` : message;
    console.error(`[${category}] ERROR: ${errorMsg}`);
  }
  
  warn(category, message) {
    console.warn(`[${category}] WARNING: ${message}`);
  }
  
  debug(category, message) {
    if (process.env.DEBUG) {
      this.log(category, `DEBUG: ${message}`);
    }
  }
  
  separator(message = '') {
    console.log('========================================');
    if (message) console.log(message);
    console.log('========================================');
  }
  
  progress(category, current, total, item = '') {
    const percentage = Math.round((current / total) * 100);
    const itemInfo = item ? ` - ${item}` : '';
    this.log(category, `Progress: ${current}/${total} (${percentage}%)${itemInfo}`);
  }
}

module.exports = Logger;
