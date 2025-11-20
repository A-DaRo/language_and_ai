const EventEmitter = require('events');
const { MESSAGE_TYPES } = require('../../src/core/ProtocolDefinitions');

/**
 * Mock WorkerProxy that simulates IPC without spawning processes
 */
class MockWorkerProxy extends EventEmitter {
  constructor(workerId = 'mock-worker-1', options = {}) {
    super();
    this.workerId = workerId;
    this.state = 'IDLE';
    this.childProcess = { pid: Math.floor(Math.random() * 10000) };
    
    // Configurable behavior
    this.responseDelay = options.responseDelay || 10; // ms
    this.shouldFail = options.shouldFail || false;
    this.crashAfter = options.crashAfter || null; // Crash after N commands
    this.commandCount = 0;
    this.customResponses = options.customResponses || {}; // Map of command types to response functions
  }
  
  sendCommand(type, payload) {
    this.state = 'BUSY';
    this.commandCount++;
    
    // Simulate crash
    if (this.crashAfter && this.commandCount >= this.crashAfter) {
      setTimeout(() => {
        this.emit('exit', { code: 1, signal: null });
      }, 5);
      return;
    }
    
    // Simulate async IPC response
    setTimeout(() => {
      if (this.shouldFail) {
        this.emit('message', {
          type: MESSAGE_TYPES.RESULT,
          data: {
            success: false,
            error: { message: 'Mock error', name: 'Error' }
          }
        });
      } else if (this.customResponses[type]) {
        // Use custom response if provided
        const response = this.customResponses[type](payload);
        this.emit('message', {
          type: MESSAGE_TYPES.RESULT,
          data: response
        });
      } else {
        // Default success response
        this.emit('message', {
          type: MESSAGE_TYPES.RESULT,
          data: {
            success: true,
            pageId: payload.pageId || 'test-page',
            resolvedTitle: 'Mock Title',
            links: []
          }
        });
      }
      this.state = 'IDLE';
    }, this.responseDelay);
  }
  
  sendInitialization(titleRegistry) {
    // Simulate initialization handshake
    setTimeout(() => {
      this.emit('message', {
        type: MESSAGE_TYPES.READY,
        data: { workerId: this.workerId }
      });
    }, 5);
  }
  
  terminate() {
    this.state = 'TERMINATED';
    this.emit('exit', { code: 0, signal: 'SIGTERM' });
  }
  
  kill(signal = 'SIGTERM') {
    this.terminate();
  }
  
  isIdle() {
    return this.state === 'IDLE';
  }
  
  isBusy() {
    return this.state === 'BUSY';
  }
}

module.exports = MockWorkerProxy;
