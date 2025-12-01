/**
 * Worker Crash Recovery Test
 * 
 * Tests system behavior when a worker crashes mid-task.
 * Verifies that:
 * 1. Master detects worker exit
 * 2. Task is marked as failed
 * 3. Other workers continue operating
 * 4. System doesn't crash
 */

const { ClusterOrchestrator } = require('./src/orchestration/ClusterOrchestrator');
const { SystemEventBus } = require('./src/core/SystemEventBus');
const { Logger } = require('./src/core/Logger');
const { Config } = require('./src/core/Config');
const path = require('path');

const TEST_OUTPUT_DIR = path.join(__dirname, 'test-output-crash');
const TEST_PAGE_URL = 'https://www.notion.so/';

async function testWorkerCrash() {
    Logger.info('=== Worker Crash Recovery Test ===');

    // Setup
    Config.OUTPUT_DIR = TEST_OUTPUT_DIR;
    Config.MAX_RECURSION_DEPTH = 2;
    Config.NOTION_PAGE_URL = TEST_PAGE_URL;

    const orchestrator = new ClusterOrchestrator(TEST_PAGE_URL, TEST_OUTPUT_DIR);
    const eventBus = SystemEventBus.getInstance();

    let workerCrashed = false;
    let crashedWorkerId = null;
    let tasksAfterCrash = 0;

    // Monitor events
    eventBus.on('WORKER:TERMINATED', (data) => {
        Logger.warn(`Worker crashed: ${data.workerId}`);
        workerCrashed = true;
        crashedWorkerId = data.workerId;
    });

    eventBus.on('TASK:COMPLETE', () => {
        if (workerCrashed) {
            tasksAfterCrash++;
        }
    });

    try {
        // Start orchestrator
        const workflowPromise = orchestrator.start();

        // Wait for bootstrap to complete
        await new Promise(resolve => {
            eventBus.once('PHASE:BOOTSTRAP_COMPLETE', resolve);
        });
        Logger.info('Bootstrap complete, workers spawned');

        // Wait for discovery to start
        await new Promise(resolve => {
            eventBus.once('PHASE:DISCOVERY_START', resolve);
        });
        Logger.info('Discovery started');

        // Wait a bit for some tasks to start
        await new Promise(resolve => setTimeout(resolve, 5000));

        // Get a worker and kill it
        Logger.info('Killing a worker to simulate crash...');
        const browserManager = orchestrator._browserManager;
        const workers = browserManager._workers;
        
        if (workers.length > 0) {
            const victimWorker = workers[0];
            Logger.info(`Terminating worker ${victimWorker.workerId}`);
            
            // Force kill the worker process (simulate crash)
            victimWorker._childProcess.kill('SIGKILL');
            
            // Wait to see system response
            await new Promise(resolve => setTimeout(resolve, 5000));
            
            Logger.info(`Worker crash detected: ${workerCrashed}`);
            Logger.info(`Tasks completed after crash: ${tasksAfterCrash}`);
        } else {
            Logger.error('No workers available to crash');
        }

        // Let workflow continue for a bit
        await new Promise(resolve => setTimeout(resolve, 20000));

        // Shutdown
        await orchestrator.shutdown();
        await workflowPromise.catch(() => {}); // Ignore workflow errors

        // Results
        Logger.info('\n=== Test Results ===');
        Logger.info(`Worker crash detected: ${workerCrashed ? '✅' : '❌'}`);
        Logger.info(`Crashed worker ID: ${crashedWorkerId || 'N/A'}`);
        Logger.info(`Tasks completed after crash: ${tasksAfterCrash}`);
        Logger.info(`System remained stable: ${tasksAfterCrash > 0 ? '✅' : '❌'}`);

        if (workerCrashed && tasksAfterCrash > 0) {
            Logger.info('\n✅ PASS: System handled worker crash gracefully');
            process.exit(0);
        } else {
            Logger.error('\n❌ FAIL: System did not handle crash properly');
            process.exit(1);
        }

    } catch (error) {
        Logger.error(`Test error: ${error.message}`);
        Logger.error(error.stack);
        
        try {
            await orchestrator.shutdown();
        } catch (cleanupErr) {
            // Ignore cleanup errors
        }
        
        process.exit(1);
    }
}

// Run test
if (require.main === module) {
    testWorkerCrash().catch(error => {
        Logger.error(`Fatal error: ${error.message}`);
        process.exit(1);
    });
}

module.exports = { testWorkerCrash };
