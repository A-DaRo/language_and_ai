const waitForEvent = (bus, eventName, timeoutMs = 1000) => {
  return new Promise((resolve, reject) => {
    const handler = (payload) => {
      clearTimeout(timeoutId);
      resolve(payload);
    };

    const timeoutId = setTimeout(() => {
      bus.off(eventName, handler);
      reject(new Error(`Event ${eventName} not emitted within ${timeoutMs}ms`));
    }, timeoutMs);

    bus.once(eventName, handler);
  });
};

module.exports = { waitForEvent };
