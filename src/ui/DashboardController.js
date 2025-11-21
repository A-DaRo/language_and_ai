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
    this.workerTaskCounts = new Map(); // Track tasks per worker
    this.startTime = Date.now();
    this.timerInterval = null;
    this.currentPhaseLabel = '';
    this.isRunning = false;
    
    // Initialize worker slot map immediately
    const workerIds = this.manager.getAllWorkerIds();
    workerIds.forEach((id, index) => this.workerSlotMap.set(id, index));
    
    // Attach listeners once
    this._attachListeners();
  }

  /**
   * @method start
   * @summary Initializes the dashboard.
   */
  start() {
    if (this.isRunning) return;
    
    // Re-initialize worker slot map in case workers changed (e.g. restart)
    const workerIds = this.manager.getAllWorkerIds();
    this.workerSlotMap.clear();
    workerIds.forEach((id, index) => this.workerSlotMap.set(id, index));

    this.dashboard = new TerminalDashboard(workerIds.length);
    this._startElapsedTimer();
    this.isRunning = true;
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
    this.isRunning = false;
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
      const timeStr = `${minutes}m ${seconds}s`;
      
      if (this.dashboard) {
        const phasePart = this.currentPhaseLabel ? ` | ${this.currentPhaseLabel}` : '';
        this.dashboard.updateHeader(` JBC090 Language & AI${phasePart} | Elapsed: ${timeStr}`);
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
      // If entering download phase and dashboard is stopped, restart it
      if (phase === 'download' && !this.isRunning) {
        this.start();
      }
      
      if (this.dashboard) {
        this.dashboard.setMode(phase, data);
        this.currentPhaseLabel = phase === 'discovery' ? 'Discovery Phase' : 'Download Phase';
        
        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        const timeStr = `${minutes}m ${seconds}s`;
        this.dashboard.updateHeader(` JBC090 Language & AI | ${this.currentPhaseLabel} | Elapsed: ${timeStr}`);
      }
    });

    this.bus.on('PHASE:STOPPING_DASHBOARD', () => {
      // Stop the dashboard and clear terminal before user confirmation
      this.stop();
      // Clear terminal with ANSI reset sequence
      process.stdout.write('\x1Bc');
      // Reset worker task counts for next phase or run
      this.workerTaskCounts.clear();
    });

    this.bus.on('DISCOVERY:PROGRESS', (stats) => {
      if (this.dashboard) {
        this.dashboard.updateDiscoveryStats(stats);
      }
    });

    this.bus.on('EXECUTION:PROGRESS', (stats) => {
      if (this.dashboard) {
        this.dashboard.updateDownloadStats(stats);
      }
    });

    this.bus.on('WORKER:BUSY', ({ workerId, task }) => {
      if (!this.dashboard) return;
      
      const slot = this.workerSlotMap.get(workerId);
      if (slot !== undefined) {
        // Increment task count for this worker
        const currentCount = (this.workerTaskCounts.get(workerId) || 0) + 1;
        this.workerTaskCounts.set(workerId, currentCount);
        
        this.dashboard.updateWorkerStatus(slot, `[BUSY] Worker ${slot + 1}: ${task.description} [PagesAssigned: ${currentCount}]`);
      }
    });

    this.bus.on('WORKER:IDLE', ({ workerId }) => {
      if (!this.dashboard) return;

      const slot = this.workerSlotMap.get(workerId);
      if (slot !== undefined) {
        this.dashboard.updateWorkerStatus(slot, `[IDLE] Worker ${slot + 1}: Waiting for task...`);
      }
    });
    
    this.bus.on('WORKER:READY', ({ workerId }) => {
      if (!this.dashboard) return;

      const slot = this.workerSlotMap.get(workerId);
      if (slot !== undefined) {
        this.dashboard.updateWorkerStatus(slot, `[IDLE] Worker ${slot + 1}: Waiting for task...`);
      }
    });
  }
}

module.exports = DashboardController;
