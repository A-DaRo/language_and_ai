/**
 * Integration Test Suite for Cluster Mode
 * 
 * Tests the complete workflow:
 * 1. Bootstrap phase - worker spawning and cookie capture
 * 2. Discovery phase - parallel metadata extraction
 * 3. Conflict resolution - duplicate detection
 * 4. Download phase - full scraping with link rewriting
 * 5. Complete phase - graceful shutdown
 */

const path = require('path');
const fs = require('fs').promises;
const { ClusterOrchestrator } = require('./src/orchestration/ClusterOrchestrator');
const { Logger } = require('./src/core/Logger');
const { Config } = require('./src/core/Config');

const TEST_OUTPUT_DIR = path.join(__dirname, 'test-output');
const TEST_PAGE_URL = 'https://www.notion.so/'; // Simple public test page

/**
 * Test utilities
 */
class TestHarness {
    constructor() {
        this.testResults = [];
        this.startTime = null;
    }

    async setup() {
        Logger.info('=== Integration Test Suite Starting ===');
        this.startTime = Date.now();

        // Clean up test output directory
        try {
            await fs.rm(TEST_OUTPUT_DIR, { recursive: true, force: true });
        } catch (err) {
            // Ignore if directory doesn't exist
        }
        await fs.mkdir(TEST_OUTPUT_DIR, { recursive: true });
        
        // Override config for testing
        Config.OUTPUT_DIR = TEST_OUTPUT_DIR;
        Config.MAX_RECURSION_DEPTH = 2; // Keep it small for testing
        Config.NOTION_PAGE_URL = TEST_PAGE_URL;
        
        Logger.info(`Test output directory: ${TEST_OUTPUT_DIR}`);
        Logger.info(`Max recursion depth: ${Config.MAX_RECURSION_DEPTH}`);
    }

    recordTest(testName, passed, details = {}) {
        const result = {
            name: testName,
            passed,
            details,
            timestamp: Date.now()
        };
        this.testResults.push(result);
        
        const status = passed ? '✅ PASS' : '❌ FAIL';
        Logger.info(`${status}: ${testName}`);
        if (!passed && details.error) {
            Logger.error(`  Error: ${details.error}`);
        }
    }

    async teardown() {
        const duration = Date.now() - this.startTime;
        const passed = this.testResults.filter(r => r.passed).length;
        const failed = this.testResults.filter(r => !r.passed).length;
        
        Logger.info('=== Test Suite Complete ===');
        Logger.info(`Duration: ${(duration / 1000).toFixed(2)}s`);
        Logger.info(`Results: ${passed} passed, ${failed} failed`);
        
        if (failed > 0) {
            Logger.error('Failed tests:');
            this.testResults.filter(r => !r.passed).forEach(r => {
                Logger.error(`  - ${r.name}`);
            });
        }

        return failed === 0;
    }
}

/**
 * Test 1: Basic Cluster Initialization
 */
async function testClusterInitialization(harness) {
    try {
        const orchestrator = new ClusterOrchestrator(Config.NOTION_PAGE_URL, Config.OUTPUT_DIR);
        
        // Verify orchestrator created successfully
        if (!orchestrator) {
            throw new Error('Orchestrator not created');
        }

        harness.recordTest('Cluster initialization', true, {
            message: 'ClusterOrchestrator instantiated successfully'
        });
    } catch (error) {
        harness.recordTest('Cluster initialization', false, {
            error: error.message
        });
    }
}

/**
 * Test 2: Worker Spawning
 */
async function testWorkerSpawning(harness) {
    const orchestrator = new ClusterOrchestrator(Config.NOTION_PAGE_URL, Config.OUTPUT_DIR);
    let workersSpawned = 0;
    let bootstrapComplete = false;

    try {
        // Hook into worker spawn events
        const { SystemEventBus } = require('./src/core/SystemEventBus');
        const eventBus = SystemEventBus.getInstance();
        
        eventBus.on('WORKER:READY', () => {
            workersSpawned++;
        });

        eventBus.on('PHASE:BOOTSTRAP_COMPLETE', () => {
            bootstrapComplete = true;
        });

        // Start orchestrator (will timeout after bootstrap for this test)
        const timeoutPromise = new Promise((resolve) => {
            setTimeout(() => {
                orchestrator.shutdown();
                resolve();
            }, 15000); // 15 seconds for bootstrap
        });

        await Promise.race([
            orchestrator.start(),
            timeoutPromise
        ]);

        // Verify workers spawned
        if (workersSpawned === 0) {
            throw new Error('No workers spawned');
        }

        if (!bootstrapComplete) {
            throw new Error('Bootstrap phase did not complete');
        }

        harness.recordTest('Worker spawning', true, {
            workersSpawned,
            message: `${workersSpawned} workers spawned successfully`
        });

    } catch (error) {
        harness.recordTest('Worker spawning', false, {
            error: error.message,
            workersSpawned
        });
    }
}

/**
 * Test 3: Discovery Phase
 */
async function testDiscoveryPhase(harness) {
    const orchestrator = new ClusterOrchestrator(Config.NOTION_PAGE_URL, Config.OUTPUT_DIR);
    let discoveryStarted = false;
    let pagesDiscovered = 0;

    try {
        const { SystemEventBus } = require('./src/core/SystemEventBus');
        const eventBus = SystemEventBus.getInstance();
        
        eventBus.on('PHASE:DISCOVERY_START', () => {
            discoveryStarted = true;
        });

        eventBus.on('TASK:COMPLETE', (data) => {
            if (data.taskType === 'DISCOVER') {
                pagesDiscovered++;
            }
        });

        // Run with timeout
        const timeoutPromise = new Promise((resolve) => {
            setTimeout(() => {
                orchestrator.shutdown();
                resolve();
            }, 30000); // 30 seconds for discovery
        });

        await Promise.race([
            orchestrator.start(),
            timeoutPromise
        ]);

        if (!discoveryStarted) {
            throw new Error('Discovery phase did not start');
        }

        if (pagesDiscovered === 0) {
            throw new Error('No pages discovered');
        }

        harness.recordTest('Discovery phase', true, {
            pagesDiscovered,
            message: `Discovered ${pagesDiscovered} pages`
        });

    } catch (error) {
        harness.recordTest('Discovery phase', false, {
            error: error.message,
            discoveryStarted,
            pagesDiscovered
        });
    }
}

/**
 * Test 4: Full End-to-End Workflow
 */
async function testFullWorkflow(harness) {
    const orchestrator = new ClusterOrchestrator(Config.NOTION_PAGE_URL, Config.OUTPUT_DIR);
    const phaseTracking = {
        bootstrap: false,
        discovery: false,
        conflictResolution: false,
        download: false,
        complete: false
    };

    try {
        const { SystemEventBus } = require('./src/core/SystemEventBus');
        const eventBus = SystemEventBus.getInstance();
        
        // Track all phases
        eventBus.on('PHASE:BOOTSTRAP_COMPLETE', () => {
            phaseTracking.bootstrap = true;
            Logger.info('  ✓ Bootstrap phase completed');
        });

        eventBus.on('PHASE:DISCOVERY_COMPLETE', () => {
            phaseTracking.discovery = true;
            Logger.info('  ✓ Discovery phase completed');
        });

        eventBus.on('PHASE:CONFLICT_RESOLUTION_COMPLETE', () => {
            phaseTracking.conflictResolution = true;
            Logger.info('  ✓ Conflict resolution completed');
        });

        eventBus.on('PHASE:DOWNLOAD_COMPLETE', () => {
            phaseTracking.download = true;
            Logger.info('  ✓ Download phase completed');
        });

        eventBus.on('PHASE:COMPLETE', () => {
            phaseTracking.complete = true;
            Logger.info('  ✓ Workflow complete');
        });

        // Run full workflow with timeout
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => {
                reject(new Error('Workflow timeout after 120 seconds'));
            }, 120000);
        });

        await Promise.race([
            orchestrator.start(),
            timeoutPromise
        ]);

        // Verify all phases completed
        const allPhasesComplete = Object.values(phaseTracking).every(v => v);
        
        if (!allPhasesComplete) {
            const incompletPhases = Object.entries(phaseTracking)
                .filter(([_, completed]) => !completed)
                .map(([phase]) => phase);
            throw new Error(`Incomplete phases: ${incompletPhases.join(', ')}`);
        }

        // Verify files were created
        const files = await fs.readdir(TEST_OUTPUT_DIR);
        if (files.length === 0) {
            throw new Error('No files created in output directory');
        }

        harness.recordTest('Full end-to-end workflow', true, {
            phases: phaseTracking,
            filesCreated: files.length,
            message: `All 5 phases completed, ${files.length} files created`
        });

    } catch (error) {
        harness.recordTest('Full end-to-end workflow', false, {
            error: error.message,
            phases: phaseTracking
        });
        
        // Ensure cleanup
        try {
            await orchestrator.shutdown();
        } catch (cleanupErr) {
            Logger.error(`Cleanup error: ${cleanupErr.message}`);
        }
    }
}

/**
 * Test 5: Graceful Shutdown
 */
async function testGracefulShutdown(harness) {
    const orchestrator = new ClusterOrchestrator(Config.NOTION_PAGE_URL, Config.OUTPUT_DIR);
    let shutdownClean = false;

    try {
        // Start orchestrator
        const workflowPromise = orchestrator.start();

        // Wait a bit for workers to spawn
        await new Promise(resolve => setTimeout(resolve, 10000));

        // Trigger shutdown
        await orchestrator.shutdown();
        shutdownClean = true;

        // Try to await the workflow (should be cancelled)
        try {
            await workflowPromise;
        } catch (err) {
            // Expected - workflow interrupted
        }

        harness.recordTest('Graceful shutdown', true, {
            message: 'Orchestrator shut down cleanly'
        });

    } catch (error) {
        harness.recordTest('Graceful shutdown', false, {
            error: error.message,
            shutdownClean
        });
    }
}

/**
 * Test 6: File System Verification
 */
async function testFileSystemOutput(harness) {
    try {
        // Check if test output directory has expected structure
        const stats = await fs.stat(TEST_OUTPUT_DIR);
        if (!stats.isDirectory()) {
            throw new Error('Output path is not a directory');
        }

        const files = await fs.readdir(TEST_OUTPUT_DIR);
        const htmlFiles = files.filter(f => f.endsWith('.html'));
        
        if (htmlFiles.length === 0) {
            throw new Error('No HTML files found in output');
        }

        // Verify at least one HTML file has content
        const sampleFile = path.join(TEST_OUTPUT_DIR, htmlFiles[0]);
        const content = await fs.readFile(sampleFile, 'utf-8');
        
        if (content.length < 100) {
            throw new Error('HTML file appears empty or truncated');
        }

        harness.recordTest('File system output', true, {
            filesCreated: files.length,
            htmlFiles: htmlFiles.length,
            message: `${htmlFiles.length} HTML files created successfully`
        });

    } catch (error) {
        harness.recordTest('File system output', false, {
            error: error.message
        });
    }
}

/**
 * Main test runner
 */
async function runTests() {
    const harness = new TestHarness();
    
    try {
        await harness.setup();

        // Run tests sequentially to avoid conflicts
        Logger.info('\n--- Test 1: Cluster Initialization ---');
        await testClusterInitialization(harness);

        Logger.info('\n--- Test 2: Worker Spawning ---');
        await testWorkerSpawning(harness);

        Logger.info('\n--- Test 3: Discovery Phase ---');
        await testDiscoveryPhase(harness);

        Logger.info('\n--- Test 4: Full End-to-End Workflow ---');
        await testFullWorkflow(harness);

        Logger.info('\n--- Test 5: Graceful Shutdown ---');
        await testGracefulShutdown(harness);

        Logger.info('\n--- Test 6: File System Verification ---');
        await testFileSystemOutput(harness);

        const success = await harness.teardown();
        process.exit(success ? 0 : 1);

    } catch (error) {
        Logger.error(`Test suite error: ${error.message}`);
        Logger.error(error.stack);
        process.exit(1);
    }
}

// Run tests if called directly
if (require.main === module) {
    runTests().catch(error => {
        Logger.error(`Fatal error: ${error.message}`);
        process.exit(1);
    });
}

module.exports = { runTests, TestHarness };
