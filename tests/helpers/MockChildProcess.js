const EventEmitter = require('events');

/**
 * Mock ChildProcess for IPC testing
 */
class MockChildProcess extends EventEmitter {
  constructor(options = {}) {
    super();
    this.pid = options.pid || Math.floor(Math.random() * 10000);
    this.connected = true;
    this.killed = false;
    this.exitCode = null;
    this.signalCode = null;
    
    // Configurable behavior
    this.shouldCrash = options.shouldCrash || false;
    this.crashDelay = options.crashDelay || 100;
    
    if (this.shouldCrash) {
      setTimeout(() => {
        this.crash();
      }, this.crashDelay);
    }
  }
  
  send(message) {
    if (!this.connected) {
      throw new Error('Channel closed');
    }
    
    // Echo back a mock response after short delay
    setTimeout(() => {
      if (this.connected) {
        this.emit('message', {
          type: 'RESULT',
          data: { success: true, mockResponse: true }
        });
      }
    }, 10);
  }
  
  kill(signal = 'SIGTERM') {
    this.killed = true;
    this.signalCode = signal;
    this.connected = false;
    
    setTimeout(() => {
      this.emit('exit', 0, signal);
    }, 10);
  }
  
  crash() {
    this.exitCode = 1;
    this.connected = false;
    this.emit('exit', 1, null);
  }
  
  disconnect() {
    this.connected = false;
    this.emit('disconnect');
  }
}

module.exports = MockChildProcess;
