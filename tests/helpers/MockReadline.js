const EventEmitter = require('events');

/**
 * Mock Readline interface for user input testing
 */
class MockReadline extends EventEmitter {
  constructor(options = {}) {
    super();
    this.responses = options.responses || [];
    this.currentIndex = 0;
    this.closed = false;
  }
  
  question(query, callback) {
    if (this.closed) {
      throw new Error('Readline interface is closed');
    }
    
    // Simulate async user input
    setTimeout(() => {
      const response = this.responses[this.currentIndex] || '';
      this.currentIndex++;
      callback(response);
    }, 10);
  }
  
  close() {
    this.closed = true;
    this.emit('close');
  }
  
  /**
   * Factory function to create mock readline with predefined answers
   */
  static create(responses = []) {
    return new MockReadline({ responses });
  }
}

module.exports = MockReadline;
