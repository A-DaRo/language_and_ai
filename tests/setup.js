const SystemEventBus = require('../src/core/SystemEventBus');

/**
 * Global test setup - runs after each test to reset the singleton bus
 */
afterEach(() => {
  if (typeof SystemEventBus._reset === 'function') {
    SystemEventBus._reset();
  }
});
