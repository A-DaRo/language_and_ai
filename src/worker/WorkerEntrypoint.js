/**
 * @fileoverview Worker Process Entry Point
 * @module worker/WorkerEntrypoint
 * @description This is the main entry point for Worker processes spawned by the Master.
 * It runs in complete isolation from Master orchestration logic.
 * 
 * **CRITICAL CONTEXT WARNING**: This code runs in a CHILD PROCESS, not the Master.
 * - NO access to Master's in-memory state (GlobalQueueManager, BrowserManager, etc.)
 * - Communication ONLY via IPC (process.send / process.on('message'))
 * - Must manage its own Puppeteer browser instance
 * - Should be stateless between tasks
 * 
 * Lifecycle:
 * 1. Launch Puppeteer browser
 * 2. Initialize TaskRunner
 * 3. Send READY signal to Master
 * 4. Listen for IPC commands (INIT, DISCOVER, DOWNLOAD, SHUTDOWN)
 * 5. Execute tasks and send RESULT back to Master
 */

const puppeteer = require('puppeteer');
const { MESSAGE_TYPES, validateMessage } = require('../core/ProtocolDefinitions');
const TaskRunner = require('./TaskRunner');

/**
 * @private
 * @type {import('puppeteer').Browser}
 * @description Puppeteer browser instance owned by this worker
 */
let browser = null;

/**
 * @private
 * @type {TaskRunner}
 * @description Task execution router
 */
let taskRunner = null;

/**
 * Initialize the worker's Puppeteer browser
 * @async
 * @private
 * @returns {Promise<import('puppeteer').Browser>} Browser instance
 * @throws {Error} If browser launch fails
 */
async function initializeBrowser() {
  try {
    browser = await puppeteer.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage', // Prevent memory issues in containers
        '--disable-accelerated-2d-canvas',
        '--no-first-run',
        '--no-zygote',
        '--disable-gpu',
        '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
      ]
    });
    
    return browser;
  } catch (error) {
    console.error('[Worker] Failed to launch browser:', error);
    throw error;
  }
}

/**
 * Setup IPC message listener
 * @function setupIPCListener
 * @description Listens for messages from the Master process and routes them to TaskRunner
 * @listens process#message
 */
function setupIPCListener() {
  process.on('message', async (message) => {
    try {
      // Validate message format
      validateMessage(message);
      
      const { type, payload } = message;
      
      // Handle shutdown command
      if (type === MESSAGE_TYPES.SHUTDOWN) {
        await shutdown('Master requested shutdown');
        return;
      }
      
      // Handle SET_COOKIES command
      if (type === MESSAGE_TYPES.SET_COOKIES) {
        if (taskRunner) {
          await taskRunner.setCookies(payload.cookies);
        }
        return;
      }
      
      // Route task commands to TaskRunner
      if (type === MESSAGE_TYPES.DISCOVER || type === MESSAGE_TYPES.DOWNLOAD) {
        if (!taskRunner) {
          throw new Error('TaskRunner not initialized');
        }
        
        const result = await taskRunner.execute(type, payload);
        process.send(result);
      }
      
    } catch (error) {
      console.error('[Worker] Error handling IPC message:', error);
      
      // Send error result back to Master
      const { serializeError } = require('../core/ProtocolDefinitions');
      process.send({
        type: MESSAGE_TYPES.RESULT,
        taskType: message.type,
        error: serializeError(error)
      });
    }
  });
}

/**
 * Graceful shutdown
 * @async
 * @private
 * @param {string} reason - Reason for shutdown
 * @returns {Promise<void>}
 */
async function shutdown(reason) {
  console.log(`[Worker] Shutting down: ${reason}`);
  
  try {
    if (browser) {
      await browser.close();
      browser = null;
    }
  } catch (error) {
    console.error('[Worker] Error during shutdown:', error);
  }
  
  process.exit(0);
}

/**
 * Handle uncaught errors
 * @private
 */
function setupErrorHandlers() {
  process.on('uncaughtException', async (error) => {
    console.error('[Worker] Uncaught exception:', error);
    await shutdown('Uncaught exception');
  });
  
  process.on('unhandledRejection', async (reason, promise) => {
    console.error('[Worker] Unhandled rejection at:', promise, 'reason:', reason);
    await shutdown('Unhandled rejection');
  });
  
  process.on('SIGTERM', async () => {
    await shutdown('SIGTERM received');
  });
  
  process.on('SIGINT', async () => {
    await shutdown('SIGINT received');
  });
}

/**
 * Main worker initialization
 * @function main
 * @async
 * @description Initializes the worker process and signals readiness to the Master
 * @returns {Promise<void>}
 */
async function main() {
  try {
    console.log(`[Worker] Starting worker process (PID: ${process.pid})`);
    
    // Setup error handlers first
    setupErrorHandlers();
    
    // Initialize browser
    console.log('[Worker] Launching browser...');
    const browserInstance = await initializeBrowser();
    
    // Initialize TaskRunner
    taskRunner = new TaskRunner(browserInstance);
    
    // Setup IPC listener
    setupIPCListener();
    
    // Signal readiness to Master
    process.send({
      type: MESSAGE_TYPES.READY,
      pid: process.pid
    });
    
    console.log('[Worker] Ready and waiting for tasks');
    
  } catch (error) {
    console.error('[Worker] Fatal initialization error:', error);
    process.exit(1);
  }
}

// Start the worker if this is the main module
if (require.main === module) {
  main();
}

module.exports = { main };
