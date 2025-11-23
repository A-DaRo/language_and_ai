/**
 * @fileoverview Master-side Handle for Worker Process
 * @module cluster/WorkerProxy
 * @description Represents a single Worker process from the Master's perspective.
 * Provides a clean interface for sending commands and tracking worker state.
 * 
 * **CRITICAL CONTEXT**: This runs in the MASTER process, not the Worker.
 * It's a proxy/handle for IPC communication with a child process.
 */

const { MESSAGE_TYPES } = require('../core/ProtocolDefinitions');
const SystemEventBus = require('../core/SystemEventBus');
const Logger = require('../core/Logger');
const { WorkerStateManager, WorkerState } = require('./proxy/WorkerStateManager');
const WorkerMessageHandler = require('./proxy/WorkerMessageHandler');
const WorkerLifecycleManager = require('./proxy/WorkerLifecycleManager');

/**
 * @class WorkerProxy
 * @classdesc Master-side handle for a Worker child process.
 * Delegates state management, message handling, and lifecycle operations.
 */
class WorkerProxy {
  /**
   * @param {string} workerId - Unique worker identifier
   * @param {ChildProcess} childProcess - Node.js child process instance
   */
  constructor(workerId, childProcess) {
    this.workerId = workerId;
    this.eventBus = SystemEventBus.getInstance();

    // Initialize component managers
    this.stateManager = new WorkerStateManager();
    this.messageHandler = new WorkerMessageHandler(workerId);
    this.lifecycleManager = new WorkerLifecycleManager(workerId, childProcess);

    this._setupListeners();
  }

  /**
   * Setup IPC and process event listeners
   * @private
   */
  _setupListeners() {
    this.lifecycleManager.setupListeners(
      (message) => this.messageHandler.handleMessage(message, this.stateManager),
      (code, signal) => this.messageHandler.handleExit(code, signal, this.stateManager),
      (error) => this.messageHandler.handleError(error, this.stateManager)
    );
  }

  /**
   * Send a command to the worker
   * @async
   * @param {string} messageType - Message type from MESSAGE_TYPES
   * @param {Object} payload - Command payload
   * @returns {Promise<void>}
   * @throws {Error} If worker is not available or send fails
   */
  async sendCommand(messageType, payload) {
    if (this.stateManager.isCrashed()) {
      throw new Error(`Worker ${this.workerId} has crashed`);
    }

    if (this.stateManager.isBusy()) {
      throw new Error(`Worker ${this.workerId} is busy`);
    }

    const task = this.stateManager.markBusy(this.workerId, messageType, payload);

    this.lifecycleManager.sendCommand(messageType, payload);

    this.eventBus.emit('TASK:STARTED', {
      workerId: this.workerId,
      taskId: task.taskId,
      taskType: messageType
    });
  }

  /**
   * Send initialization payload with titleRegistry to worker
   * @async
   * @param {Object} titleRegistry - ID-to-title map
   * @returns {Promise<void>}
   */
  async sendInitialization(titleRegistry) {
    if (this.stateManager.isCrashed()) {
      Logger.getInstance().warn('WorkerProxy', `Cannot initialize crashed worker ${this.workerId}`);
      return;
    }

    this.lifecycleManager.sendInitialization(titleRegistry);
  }

  /**
   * Broadcast cookies to this worker
   * @async
   * @param {Array<Object>} cookies - Cookie objects
   * @returns {Promise<void>}
   */
  async broadcastCookies(cookies) {
    if (this.stateManager.isCrashed()) {
      Logger.getInstance().warn('WorkerProxy', `Cannot send cookies to crashed worker ${this.workerId}`);
      return;
    }

    this.lifecycleManager.broadcastCookies(cookies);
  }

  /**
   * Terminate the worker process
   * @async
   * @returns {Promise<void>}
   */
  async terminate() {
    await this.lifecycleManager.terminate();
  }

  /**
   * Check if worker is available for tasks
   * @returns {boolean} True if worker is idle and ready
   */
  isAvailable() {
    return this.stateManager.isAvailable();
  }

  /**
   * Get worker status
   * @returns {Object} Status object
   */
  getStatus() {
    return {
      workerId: this.workerId,
      state: this.stateManager.getState(),
      pid: this.lifecycleManager.getPid(),
      currentTask: this.stateManager.getCurrentTask()
    };
  }
}

module.exports = { WorkerProxy, WorkerState };
