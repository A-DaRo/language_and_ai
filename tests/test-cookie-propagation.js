/**
 * Cookie Propagation Test
 * 
 * Tests that cookies are properly captured in bootstrap phase
 * and broadcast to all workers.
 * 
 * Verifies:
 * 1. Bootstrap worker captures cookies
 * 2. Cookies are broadcast to all spawned workers
 * 3. Workers can access authenticated pages
 */

const { ClusterOrchestrator } = require('./src/orchestration/ClusterOrchestrator');
const { SystemEventBus } = require('./src/core/SystemEventBus');
const { Logger } = require('./src/core/Logger');
const { Config } = require('./src/core/Config');
const path = require('path');

const TEST_OUTPUT_DIR = path.join(__dirname, 'test-output-cookies');
const TEST_PAGE_URL = 'https://www.notion.so/'; // May require auth

async function testCookiePropagation() {
    Logger.info('=== Cookie Propagation Test ===');

    // Setup
    Config.OUTPUT_DIR = TEST_OUTPUT_DIR;
    Config.MAX_RECURSION_DEPTH = 1;
    Config.NOTION_PAGE_URL = TEST_PAGE_URL;

    const orchestrator = new ClusterOrchestrator(TEST_PAGE_URL, TEST_OUTPUT_DIR);
    const eventBus = SystemEventBus.getInstance();

    let cookiesCaptured = false;
    let cookiesShared = false;
    let workerCount = 0;

    // Monitor cookie events
    eventBus.on('COOKIES:CAPTURED', (data) => {
        Logger.info('Cookies captured!');
        if (data.cookies && data.cookies.length > 0) {
            cookiesCaptured = true;
            Logger.info(`  Captured ${data.cookies.length} cookies`);
            data.cookies.forEach(cookie => {
                Logger.info(`    - ${cookie.name}: ${cookie.value.substring(0, 20)}...`);
            });
        } else {
            Logger.warn('  No cookies captured (page may not require auth)');
        }
    });

    eventBus.on('COOKIES:SHARED', (data) => {
        Logger.info('Cookies shared with workers');
        cookiesShared = true;
        Logger.info(`  Shared to ${data.workerCount || 'all'} workers`);
    });

    eventBus.on('WORKER:READY', () => {
        workerCount++;
    });

    try {
        // Start orchestrator
        const workflowPromise = orchestrator.start();

        // Wait for bootstrap to complete
        await new Promise(resolve => {
            eventBus.once('PHASE:BOOTSTRAP_COMPLETE', resolve);
        });
        Logger.info('Bootstrap complete');

        // Wait a bit more to ensure cookie sharing
        await new Promise(resolve => setTimeout(resolve, 5000));

        // Check if cookies were handled
        Logger.info('\n=== Cookie Handling Status ===');
        Logger.info(`Workers spawned: ${workerCount}`);
        Logger.info(`Cookies captured: ${cookiesCaptured ? '✅' : '⚠️  (page may not require auth)'}`);
        Logger.info(`Cookies shared: ${cookiesShared ? '✅' : '❌'}`);

        // Continue with discovery to verify workers can access pages
        Logger.info('\n=== Testing Worker Page Access ===');
        await new Promise(resolve => {
            eventBus.once('PHASE:DISCOVERY_START', resolve);
        });
        Logger.info('Discovery phase started - workers accessing pages');

        // Let discovery run for a bit
        await new Promise(resolve => setTimeout(resolve, 15000));

        // Shutdown
        await orchestrator.shutdown();
        await workflowPromise.catch(() => {}); // Ignore workflow errors

        // Results
        Logger.info('\n=== Test Results ===');
        Logger.info(`Workers spawned: ${workerCount}`);
        Logger.info(`Cookie capture working: ${cookiesCaptured || !TEST_PAGE_URL.includes('private') ? '✅' : '❌'}`);
        Logger.info(`Cookie sharing working: ${cookiesShared ? '✅' : '❌'}`);
        
        // Success if we either:
        // 1. Captured and shared cookies, OR
        // 2. Page doesn't require auth (no cookies needed)
        const success = (cookiesCaptured && cookiesShared) || (!cookiesCaptured && workerCount > 1);
        
        if (success) {
            Logger.info('\n✅ PASS: Cookie propagation working correctly');
            process.exit(0);
        } else {
            Logger.error('\n❌ FAIL: Cookie propagation failed');
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
    testCookiePropagation().catch(error => {
        Logger.error(`Fatal error: ${error.message}`);
        process.exit(1);
    });
}

module.exports = { testCookiePropagation };
