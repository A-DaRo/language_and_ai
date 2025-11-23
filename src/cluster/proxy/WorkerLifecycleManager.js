/**
 * @fileoverview Worker Lifecycle Manager for WorkerProxy
 * @module cluster/proxy/WorkerLifecycleManager
 * @description Manages worker process lifecycle (startup, shutdown, stream handling)
 */

const { MESSAGE_TYPES } = require('../../core/ProtocolDefinitions');
const Logger = require('../../core/Logger');

/**
 * @class WorkerLifecycleManager
 * @classdesc Handles worker initialization, stream setup, and termination
 */
class WorkerLifecycleManager {
  constructor(workerId, childProcess) {
    this.workerId = workerId;
    this.childProcess = childProcess;
    this.logger = Logger.getInstance();
  }

  /**
   * Setup IPC listeners on child process
   * @param {Function} messageHandler - Handler for incoming messages
   * @param {Function} exitHandler - Handler for process exit
   * @param {Function} errorHandler - Handler for process error
   * @returns {void}
   */
  setupListeners(messageHandler, exitHandler, errorHandler) {
    // Listen for messages
    this.childProcess.on('message', messageHandler);

    // Capture stdout and route through Logger
    if (this.childProcess.stdout) {
      this.childProcess.stdout.on('data', (data) => {
        this.logger.info(`Worker-${this.workerId}:stdout`, data.toString().trim());
      });
    }

    // Capture stderr and route through Logger
    if (this.childProcess.stderr) {
      this.childProcess.stderr.on('data', (data) => {
        this.logger.error(`Worker-${this.workerId}:stderr`, data.toString().trim());
      });
    }

    // Handle exit
    this.childProcess.on('exit', exitHandler);

    // Handle errors
    this.childProcess.on('error', errorHandler);
  }

  /**
   * Send a command to the worker
   * @param {string} messageType - Message type
   * @param {Object} payload - Command payload
   * @returns {void}
   * @throws {Error} If send fails
   */
  sendCommand(messageType, payload) {
    this.childProcess.send({
      type: messageType,
      payload: payload
    });
  }

  /**
   * Send initialization message
   * @param {Object} titleRegistry - ID-to-title map
   * @returns {void}
   */
  sendInitialization(titleRegistry) {
    this.sendCommand(MESSAGE_TYPES.INIT, {
      titleRegistry: titleRegistry || {}
    });
  }

  /**
   * Broadcast cookies to worker
   * @param {Array<Object>} cookies - Cookie objects
   * @returns {void}
   */
  broadcastCookies(cookies) {
    this.sendCommand(MESSAGE_TYPES.SET_COOKIES, {
      cookies: cookies
    });
  }

  /**
   * Terminate the worker process
   * @async
   * @returns {Promise<void>}
   */
  async terminate() {
    this.logger.info('WorkerLifecycleManager', `Terminating worker ${this.workerId}`);

    try {
      // Send shutdown command
      this.sendCommand(MESSAGE_TYPES.SHUTDOWN, {});

      // Wait for graceful shutdown
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Force kill if still alive
      if (!this.childProcess.killed) {
        this.childProcess.kill('SIGTERM');
      }
    } catch (error) {
      this.logger.error('WorkerLifecycleManager', 'Error terminating worker', error);
      this.childProcess.kill('SIGKILL');
    }
  }

  /**
   * Get process PID
   * @returns {number}
   */
  getPid() {
    return this.childProcess.pid;
  }
}

module.exports = WorkerLifecycleManager;
