/**
 * @fileoverview UI Logic Coordinator
 * @module ui/DashboardController
 */

const TerminalDashboard = require('./TerminalDashboard');

/**
 * @class DashboardController
 * @description The "brain" of the UI. It translates system events into visual
 * updates on the dashboard, keeping the rendering logic simple.
 */
class DashboardController {
  /**
   * @constructor
   * @param {SystemEventBus} eventBus - The master event bus.
   * @param {BrowserManager} browserManager - For mapping worker IDs to slots.
   */
  constructor(eventBus, browserManager) {
    this.bus = eventBus;
    this.manager = browserManager;
    this.dashboard = null;
    this.workerSlotMap = new Map();
    this.startTime = Date.now();
    this.timerInterval = null;
  }

  /**
   * @method start
   * @summary Initializes the dashboard and attaches all event listeners.
   */
  start() {
    const workerIds = this.manager.getAllWorkerIds();
    workerIds.forEach((id, index) => this.workerSlotMap.set(id, index));

    this.dashboard = new TerminalDashboard(workerIds.length);
    this._attachListeners();
    this._startElapsedTimer();
  }

  /**
   * @method getDashboard
   * @returns {TerminalDashboard} The raw dashboard instance for the logger.
   */
  getDashboard() {
    return this.dashboard;
  }
  
  /**
   * @method stop
   * @description Stops the dashboard renderer and timer.
   */
  stop() {
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
    if (this.dashboard) {
      this.dashboard.stop();
    }
  }

  /**
   * @private
   * @method _startElapsedTimer
   * @description Starts a timer to update the elapsed time in the header.
   */
  _startElapsedTimer() {
    this.timerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
      const minutes = Math.floor(elapsed / 60);
      const seconds = elapsed % 60;
      const timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}`;
      
      if (this.dashboard) {
        this.dashboard.updateHeader(`JBC090 Language & AI | Elapsed: ${timeStr}`);
      }
    }, 1000);
  }

  /**
   * @private
   * @method _attachListeners
   * @description Subscribes to all relevant SystemEventBus events.
   */
  _attachListeners() {
    this.bus.on('PHASE:CHANGED', ({ phase, data }) => {
      this.dashboard.setMode(phase, data);
      const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
      const minutes = Math.floor(elapsed / 60);
      const seconds = elapsed % 60;
      const timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}`;
      this.dashboard.updateHeader(`JBC090 Language & AI | ${phase.toUpperCase()} Phase | Elapsed: ${timeStr}`);
    });

    this.bus.on('DISCOVERY:PROGRESS', (stats) => {
      this.dashboard.updateDiscoveryStats(stats);
    });

    this.bus.on('EXECUTION:PROGRESS', (stats) => {
      this.dashboard.updateDownloadStats(stats);
    });

    this.bus.on('WORKER:BUSY', ({ workerId, task }) => {
      const slot = this.workerSlotMap.get(workerId);
      if (slot !== undefined) {
        this.dashboard.updateWorkerStatus(slot, `[BUSY] ${task.description}`);
      }
    });

    this.bus.on('WORKER:IDLE', ({ workerId }) => {
      const slot = this.workerSlotMap.get(workerId);
      if (slot !== undefined) {
        this.dashboard.updateWorkerStatus(slot, `[IDLE] Waiting for task...`);
      }
    });
    
    this.bus.on('WORKER:READY', ({ workerId }) => {
      const slot = this.workerSlotMap.get(workerId);
      if (slot !== undefined) {
        this.dashboard.updateWorkerStatus(slot, `[IDLE] Waiting for task...`);
      }
    });
  }
}

module.exports = DashboardController;
