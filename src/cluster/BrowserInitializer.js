/**
 * @fileoverview Worker Pool Bootstrap Logic
 * @module cluster/BrowserInitializer
 * @description Handles the physical spawning of Worker processes with capacity planning.
 * Determines optimal worker count based on system resources.
 */

const { fork } = require('child_process');
const os = require('os');
const path = require('path');
const { WorkerProxy } = require('./WorkerProxy');

/**
 * @class BrowserInitializer
 * @classdesc Factory for spawning worker processes with resource-aware capacity planning
 */
class BrowserInitializer {
  /**
   * Calculate optimal worker count based on system resources
   * @static
   * @param {number} [minWorkers=2] - Minimum number of workers
   * @param {number} [maxWorkers=8] - Maximum number of workers
   * @returns {Object} Capacity information
   * @returns {number} return.workerCount - Recommended worker count
   * @returns {number} return.totalMemoryMB - Total system memory in MB
   * @returns {number} return.freeMemoryMB - Free system memory in MB
   * @returns {number} return.cpuCount - Number of CPU cores
   * @example
   * const capacity = BrowserInitializer.calculateCapacity();
   * // Returns: { workerCount: 4, totalMemoryMB: 8192, freeMemoryMB: 4096, cpuCount: 8 }
   */
  static calculateCapacity(minWorkers = 2, maxWorkers = 8) {
    const totalMemory = os.totalmem();
    const freeMemory = os.freemem();
    const cpuCount = os.cpus().length;
    
    const totalMemoryMB = Math.round(totalMemory / (1024 * 1024));
    const freeMemoryMB = Math.round(freeMemory / (1024 * 1024));
    
    // Assume each worker needs ~1GB RAM for Puppeteer browser
    const MEMORY_PER_WORKER_MB = 1024;
    
    // Calculate worker count based on memory
    const memoryBasedWorkers = Math.floor(freeMemoryMB * 0.7 / MEMORY_PER_WORKER_MB);
    
    // Don't exceed CPU count
    const cpuBasedWorkers = Math.max(1, cpuCount - 1);
    
    // Take the minimum of memory and CPU constraints
    let workerCount = Math.min(memoryBasedWorkers, cpuBasedWorkers);
    
    // Enforce min/max bounds
    workerCount = Math.max(minWorkers, Math.min(maxWorkers, workerCount));
    
    return {
      workerCount,
      totalMemoryMB,
      freeMemoryMB,
      cpuCount
    };
  }
  
  /**
   * Spawn a pool of worker processes
   * @static
   * @async
   * @param {number} count - Number of workers to spawn
   * @param {number} [startIndex=1] - Starting index for worker IDs
   * @returns {Promise<Array<WorkerProxy>>} Array of worker proxies
   * @throws {Error} If worker spawning fails
   * @example
   * const workers = await BrowserInitializer.spawnWorkerPool(4);
   * console.log(`Spawned ${workers.length} workers`);
   */
  static async spawnWorkerPool(count, startIndex = 1) {
    console.log(`[BrowserInitializer] Spawning ${count} worker process(es)...`);
    
    const workers = [];
    const promises = [];
    
    for (let i = 0; i < count; i++) {
      const promise = BrowserInitializer._spawnSingleWorker(startIndex + i);
      promises.push(promise);
    }
    
    const results = await Promise.all(promises);
    workers.push(...results);
    
    console.log(`[BrowserInitializer] Successfully spawned ${workers.length} worker(s)`);
    return workers;
  }
  
  /**
   * Spawn a single worker process
   * @private
   * @static
   * @async
   * @param {number} index - Worker index (for ID generation)
   * @returns {Promise<WorkerProxy>} Worker proxy
   * @throws {Error} If worker spawn fails or doesn't send READY within timeout
   */
  static async _spawnSingleWorker(index) {
    const workerId = `worker-${index}`;
    const workerScriptPath = path.join(__dirname, '..', 'worker', 'WorkerEntrypoint.js');
    
    console.log(`[BrowserInitializer] Spawning worker ${workerId}...`);
    
    // Spawn child process
    const childProcess = fork(workerScriptPath, [], {
      silent: false, // Inherit stdio for logs
      env: { ...process.env, WORKER_ID: workerId }
    });
    
    // Create proxy
    const proxy = new WorkerProxy(workerId, childProcess);
    
    // Wait for READY signal with timeout
    const readyPromise = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Worker ${workerId} did not send READY within 30 seconds`));
      }, 30000);
      
      const readyHandler = (event) => {
        if (event.workerId === workerId) {
          clearTimeout(timeout);
          proxy.eventBus.off('WORKER:READY', readyHandler);
          resolve(proxy);
        }
      };
      
      proxy.eventBus.on('WORKER:READY', readyHandler);
    });
    
    return readyPromise;
  }
}

module.exports = BrowserInitializer;
