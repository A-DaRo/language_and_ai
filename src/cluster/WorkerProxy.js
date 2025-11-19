/**
 * @fileoverview Master-side Handle for Worker Process
 * @module cluster/WorkerProxy
 * @description Represents a single Worker process from the Master's perspective.
 * Provides a clean interface for sending commands and tracking worker state.
 * 
 * **CRITICAL CONTEXT**: This runs in the MASTER process, not the Worker.
 * It's a proxy/handle for IPC communication with a child process.
 */

const { MESSAGE_TYPES, validateMessage } = require('../core/ProtocolDefinitions');
const SystemEventBus = require('../core/SystemEventBus');

/**
 * Worker state constants
 * @enum {string}
 */
const WorkerState = {
  INITIALIZING: 'INITIALIZING',
  IDLE: 'IDLE',
  BUSY: 'BUSY',
  CRASHED: 'CRASHED'
};

/**
 * @class WorkerProxy
 * @classdesc Master-side handle for a Worker child process.
 * Manages IPC communication, state tracking, and lifecycle.
 */
class WorkerProxy {
  /**
   * @param {string} workerId - Unique worker identifier
   * @param {ChildProcess} childProcess - Node.js child process instance
   */
  constructor(workerId, childProcess) {
    this.workerId = workerId;
    this.childProcess = childProcess;
    this.state = WorkerState.INITIALIZING;
    this.currentTask = null;
    this.eventBus = SystemEventBus.getInstance();
    
    this._setupListeners();
  }
  
  /**
   * Setup IPC and process event listeners
   * @private
   */
  _setupListeners() {
    // Listen for messages from worker
    this.childProcess.on('message', (message) => {
      this._handleMessage(message);
    });
    
    // Handle worker exit
    this.childProcess.on('exit', (code, signal) => {
      this._handleExit(code, signal);
    });
    
    // Handle worker errors
    this.childProcess.on('error', (error) => {
      console.error(`[WorkerProxy] Worker ${this.workerId} error:`, error);
      this.state = WorkerState.CRASHED;
      this.eventBus.emit('WORKER:CRASHED', {
        workerId: this.workerId,
        error: error.message
      });
    });
  }
  
  /**
   * Handle IPC message from worker
   * @private
   * @param {Object} message - IPC message
   */
  _handleMessage(message) {
    try {
      validateMessage(message);
      
      if (message.type === MESSAGE_TYPES.READY) {
        this._handleReady(message);
      } else if (message.type === MESSAGE_TYPES.RESULT) {
        this._handleResult(message);
      } else {
        console.warn(`[WorkerProxy] Unknown message type from worker: ${message.type}`);
      }
    } catch (error) {
      console.error(`[WorkerProxy] Error handling message:`, error);
    }
  }
  
  /**
   * Handle READY signal from worker
   * @private
   * @param {Object} message - Ready message
   */
  _handleReady(message) {
    console.log(`[WorkerProxy] Worker ${this.workerId} is ready (PID: ${message.pid})`);
    this.state = WorkerState.IDLE;
    this.eventBus.emit('WORKER:READY', {
      workerId: this.workerId,
      pid: message.pid
    });
  }
  
  /**
   * Handle task result from worker
   * @private
   * @param {Object} message - Result message
   */
  _handleResult(message) {
    console.log(`[WorkerProxy] Worker ${this.workerId} completed task`);
    
    const taskId = this.currentTask ? this.currentTask.taskId : null;
    
    if (message.error) {
      // Task failed
      this.eventBus.emit('TASK:FAILED', {
        workerId: this.workerId,
        taskId: taskId,
        taskType: message.taskType,
        error: message.error
      });
    } else {
      // Task succeeded
      this.eventBus.emit('TASK:COMPLETE', {
        workerId: this.workerId,
        taskId: taskId,
        taskType: message.taskType,
        result: message.data
      });
    }
    
    // Mark worker as idle
    this.currentTask = null;
    this.state = WorkerState.IDLE;
    this.eventBus.emit('WORKER:IDLE', {
      workerId: this.workerId
    });
  }
  
  /**
   * Handle worker process exit
   * @private
   * @param {number} code - Exit code
   * @param {string} signal - Exit signal
   */
  _handleExit(code, signal) {
    console.log(`[WorkerProxy] Worker ${this.workerId} exited (code: ${code}, signal: ${signal})`);
    this.state = WorkerState.CRASHED;
    
    this.eventBus.emit('WORKER:CRASHED', {
      workerId: this.workerId,
      exitCode: code,
      signal: signal
    });
  }
  
  /**
   * Send a command to the worker
   * @async
   * @param {string} messageType - Message type from MESSAGE_TYPES
   * @param {Object} payload - Command payload
   * @returns {Promise<void>}
   * @throws {Error} If worker is not available or send fails
   * @example
   * await workerProxy.sendCommand(MESSAGE_TYPES.DISCOVER, {
   *   url: 'https://notion.so/page',
   *   pageId: 'abc123',
   *   depth: 1
   * });
   */
  async sendCommand(messageType, payload) {
    if (this.state === WorkerState.CRASHED) {
      throw new Error(`Worker ${this.workerId} has crashed`);
    }
    
    if (this.state === WorkerState.BUSY) {
      throw new Error(`Worker ${this.workerId} is busy`);
    }
    
    this.state = WorkerState.BUSY;
    this.currentTask = {
      taskId: `${this.workerId}-${Date.now()}`,
      type: messageType,
      payload: payload
    };
    
    this.childProcess.send({
      type: messageType,
      payload: payload
    });
    
    this.eventBus.emit('TASK:STARTED', {
      workerId: this.workerId,
      taskId: this.currentTask.taskId,
      taskType: messageType
    });
  }
  
  /**
   * Broadcast cookies to this worker
   * @async
   * @param {Array<Object>} cookies - Cookie objects
   * @returns {Promise<void>}
   */
  async broadcastCookies(cookies) {
    if (this.state === WorkerState.CRASHED) {
      console.warn(`[WorkerProxy] Cannot send cookies to crashed worker ${this.workerId}`);
      return;
    }
    
    this.childProcess.send({
      type: MESSAGE_TYPES.SET_COOKIES,
      payload: { cookies }
    });
  }
  
  /**
   * Terminate the worker process
   * @async
   * @returns {Promise<void>}
   */
  async terminate() {
    console.log(`[WorkerProxy] Terminating worker ${this.workerId}`);
    
    try {
      // Send shutdown command
      this.childProcess.send({
        type: MESSAGE_TYPES.SHUTDOWN
      });
      
      // Wait a bit for graceful shutdown
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Force kill if still alive
      if (!this.childProcess.killed) {
        this.childProcess.kill('SIGTERM');
      }
    } catch (error) {
      console.error(`[WorkerProxy] Error terminating worker:`, error);
      this.childProcess.kill('SIGKILL');
    }
  }
  
  /**
   * Check if worker is available for tasks
   * @returns {boolean} True if worker is idle and ready
   */
  isAvailable() {
    return this.state === WorkerState.IDLE;
  }
  
  /**
   * Get worker status
   * @returns {Object} Status object
   */
  getStatus() {
    return {
      workerId: this.workerId,
      state: this.state,
      pid: this.childProcess.pid,
      currentTask: this.currentTask
    };
  }
}

module.exports = { WorkerProxy, WorkerState };
