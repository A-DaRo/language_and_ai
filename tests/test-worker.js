/**
 * Test script for Worker process
 * Tests that a worker can spawn, initialize, and respond to IPC
 */

const { fork } = require('child_process');
const path = require('path');

console.log('Spawning worker process...');

const workerPath = path.join(__dirname, 'src', 'worker', 'WorkerEntrypoint.js');
const worker = fork(workerPath);

worker.on('message', (message) => {
  console.log('✓ Received message from worker:', JSON.stringify(message, null, 2));
  
  if (message.type === 'IPC_READY') {
    console.log('✓ Worker is ready! Test passed.');
    worker.kill();
  }
});

worker.on('error', (error) => {
  console.error('✗ Worker error:', error);
  process.exit(1);
});

worker.on('exit', (code, signal) => {
  console.log(`Worker exited with code ${code} and signal ${signal}`);
  process.exit(code === 0 ? 0 : 1);
});

// Timeout after 15 seconds
setTimeout(() => {
  console.error('✗ Test timeout - worker did not become ready');
  worker.kill();
  process.exit(1);
}, 15000);
