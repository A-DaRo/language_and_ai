/**
 * @fileoverview Worker State Management for WorkerProxy
 * @module cluster/proxy/WorkerStateManager
 * @description Manages worker process state transitions and state validation
 */

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
 * @class WorkerStateManager
 * @classdesc Manages state transitions and provides state validation
 */
class WorkerStateManager {
  constructor() {
    this.state = WorkerState.INITIALIZING;
    this.currentTask = null;
  }

  /**
   * Transition to IDLE state
   * @returns {void}
   */
  markIdle() {
    this.state = WorkerState.IDLE;
    this.currentTask = null;
  }

  /**
   * Transition to BUSY state with task tracking
   * @param {string} workerId - Worker ID
   * @param {string} messageType - Message type
   * @param {Object} payload - Command payload
   * @returns {Object} Task object with taskId
   */
  markBusy(workerId, messageType, payload) {
    this.state = WorkerState.BUSY;
    this.currentTask = {
      taskId: `${workerId}-${Date.now()}`,
      type: messageType,
      payload: payload
    };
    return this.currentTask;
  }

  /**
   * Transition to CRASHED state
   * @returns {void}
   */
  markCrashed() {
    this.state = WorkerState.CRASHED;
    this.currentTask = null;
  }

  /**
   * Check if worker is available
   * @returns {boolean} True if idle and not crashed
   */
  isAvailable() {
    return this.state === WorkerState.IDLE;
  }

  /**
   * Check if worker is busy
   * @returns {boolean}
   */
  isBusy() {
    return this.state === WorkerState.BUSY;
  }

  /**
   * Check if worker has crashed
   * @returns {boolean}
   */
  isCrashed() {
    return this.state === WorkerState.CRASHED;
  }

  /**
   * Get current state
   * @returns {string}
   */
  getState() {
    return this.state;
  }

  /**
   * Get current task (if any)
   * @returns {Object|null}
   */
  getCurrentTask() {
    return this.currentTask;
  }
}

module.exports = { WorkerStateManager, WorkerState };
