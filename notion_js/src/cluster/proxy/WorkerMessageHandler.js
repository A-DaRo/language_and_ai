/**
 * @fileoverview Worker Message Handler for WorkerProxy
 * @module cluster/proxy/WorkerMessageHandler
 * @description Handles IPC message processing and event emission
 */

const { MESSAGE_TYPES, validateMessage } = require('../../core/ProtocolDefinitions');
const SystemEventBus = require('../../core/SystemEventBus');
const Logger = require('../../core/Logger');

/**
 * @class WorkerMessageHandler
 * @classdesc Processes messages from worker and emits corresponding events
 */
class WorkerMessageHandler {
  constructor(workerId) {
    this.workerId = workerId;
    this.eventBus = SystemEventBus.getInstance();
    this.logger = Logger.getInstance();
  }

  /**
   * Handle IPC message from worker
   * @param {Object} message - IPC message
   * @param {Object} stateManager - Worker state manager
   * @returns {void}
   */
  handleMessage(message, stateManager) {
    try {
      // Handle log forwarding specially
      if (message.type === 'IPC_LOG') {
        this._forwardLog(message);
        return;
      }

      validateMessage(message);

      if (message.type === MESSAGE_TYPES.READY) {
        this._handleReady(message);
      } else if (message.type === MESSAGE_TYPES.RESULT) {
        this._handleResult(message, stateManager);
      } else {
        this.logger.warn('WorkerMessageHandler', `Unknown message type: ${message.type}`);
      }
    } catch (error) {
      this.logger.error('WorkerMessageHandler', 'Error handling message', error);
    }
  }

  /**
   * Forward worker log to Master Logger
   * @private
   * @param {Object} message - Log message
   * @returns {void}
   */
  _forwardLog(message) {
    const { level, category, message: logMsg, meta } = message.payload;
    const workerCategory = `Worker-${this.workerId}:${category}`;

    if (typeof this.logger[level] === 'function') {
      this.logger[level](workerCategory, logMsg, meta);
    } else {
      this.logger.info(workerCategory, logMsg, meta);
    }
  }

  /**
   * Handle READY signal from worker
   * @private
   * @param {Object} message - Ready message
   * @returns {void}
   */
  _handleReady(message) {
    this.logger.info('WorkerMessageHandler', `Worker ${this.workerId} is ready (PID: ${message.pid})`);

    this.eventBus.emit('WORKER:READY', {
      workerId: this.workerId,
      pid: message.pid
    });
  }

  /**
   * Handle task result from worker
   * @private
   * @param {Object} message - Result message
   * @param {Object} stateManager - Worker state manager
   * @returns {void}
   */
  _handleResult(message, stateManager) {
    this.logger.info('WorkerMessageHandler', `Worker ${this.workerId} completed task`);

    const taskId = stateManager.getCurrentTask() ? stateManager.getCurrentTask().taskId : null;

    if (message.error) {
      this.eventBus.emit('TASK:FAILED', {
        workerId: this.workerId,
        taskId: taskId,
        taskType: message.taskType,
        error: message.error
      });
    } else {
      this.eventBus.emit('TASK:COMPLETE', {
        workerId: this.workerId,
        taskId: taskId,
        taskType: message.taskType,
        result: message.data
      });
    }

    stateManager.markIdle();
    this.eventBus.emit('WORKER:IDLE', {
      workerId: this.workerId
    });
  }

  /**
   * Handle worker process exit
   * @param {number} code - Exit code
   * @param {string} signal - Exit signal
   * @param {Object} stateManager - Worker state manager
   * @returns {void}
   */
  handleExit(code, signal, stateManager) {
    this.logger.info('WorkerMessageHandler', `Worker ${this.workerId} exited (code: ${code}, signal: ${signal})`);

    stateManager.markCrashed();
    this.eventBus.emit('WORKER:CRASHED', {
      workerId: this.workerId,
      exitCode: code,
      signal: signal
    });
  }

  /**
   * Handle worker process error
   * @param {Error} error - Worker error
   * @param {Object} stateManager - Worker state manager
   * @returns {void}
   */
  handleError(error, stateManager) {
    this.logger.error('WorkerMessageHandler', `Worker ${this.workerId} error: ${error.message}`, error);

    stateManager.markCrashed();
    this.eventBus.emit('WORKER:CRASHED', {
      workerId: this.workerId,
      error: error.message
    });
  }
}

module.exports = WorkerMessageHandler;
