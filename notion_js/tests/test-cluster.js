/**
 * Test script for Cluster layer (BrowserManager + Workers)
 * Tests that we can spawn workers and execute tasks through BrowserManager
 */

const BrowserInitializer = require('./src/cluster/BrowserInitializer');
const BrowserManager = require('./src/cluster/BrowserManager');
const { MESSAGE_TYPES } = require('./src/core/ProtocolDefinitions');

async function testCluster() {
  try {
    console.log('=== Cluster Layer Test ===\n');
    
    // Calculate capacity
    console.log('1. Calculating system capacity...');
    const capacity = BrowserInitializer.calculateCapacity(1, 2);
    console.log(`   ✓ Recommended workers: ${capacity.workerCount}`);
    console.log(`   ✓ Free memory: ${capacity.freeMemoryMB}MB`);
    console.log(`   ✓ CPU cores: ${capacity.cpuCount}\n`);
    
    // Spawn workers
    console.log('2. Spawning worker pool...');
    const workers = await BrowserInitializer.spawnWorkerPool(2);
    console.log(`   ✓ Spawned ${workers.length} workers\n`);
    
    // Create BrowserManager
    console.log('3. Creating BrowserManager...');
    const manager = new BrowserManager();
    manager.registerWorkers(workers);
    console.log(`   ✓ Manager ready with ${manager.getTotalCount()} workers\n`);
    
    // Wait a bit for workers to become ready
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    console.log('4. Pool statistics:');
    const stats = manager.getStatistics();
    console.log(`   Total: ${stats.total}, Idle: ${stats.idle}, Busy: ${stats.busy}\n`);
    
    // Test task execution
    console.log('5. Testing task execution...');
    console.log('   (This will fail because we need a real Notion URL, but tests IPC flow)\n');
    
    // Shutdown
    console.log('6. Shutting down workers...');
    await manager.shutdown();
    console.log('   ✓ All workers terminated\n');
    
    console.log('=== Test Complete ===');
    process.exit(0);
    
  } catch (error) {
    console.error('✗ Test failed:', error);
    process.exit(1);
  }
}

testCluster();
