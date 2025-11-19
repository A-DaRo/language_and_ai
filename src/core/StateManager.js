const PageContext = require('../domain/PageContext');

/**
 * Centralized coordinator for discovery state.
 * Maintains a unique PageContext per URL and orchestrates level queues
 * for the level-synchronous crawl.
 */
class StateManager {
  constructor() {
    this.reset();
  }

  static getInstance() {
    if (!StateManager.instance) {
      StateManager.instance = new StateManager();
    }
    return StateManager.instance;
  }

  reset(rootUrl = null, rootTitle = 'Main_Page') {
    this.urlToContextMap = new Map();
    this.currentLevelQueue = [];
    this.nextLevelQueue = [];
    this.rootContext = null;

    if (rootUrl) {
      this.bootstrap(rootUrl, rootTitle);
    }
  }

  bootstrap(rootUrl, rootTitle = 'Main_Page') {
    this.urlToContextMap.clear();
    this.currentLevelQueue = [];
    this.nextLevelQueue = [];

    const rootContext = new PageContext(rootUrl, rootTitle, 0, null);
    rootContext.setDisplayTitle(rootTitle);
    this.rootContext = rootContext;
    this.urlToContextMap.set(rootUrl, rootContext);
    this.currentLevelQueue.push(rootUrl);
    return rootContext;
  }

  getRootContext() {
    return this.rootContext;
  }

  getContextByUrl(url) {
    return this.urlToContextMap.get(url);
  }

  getCurrentLevelQueue() {
    return [...this.currentLevelQueue];
  }

  hasCurrentLevelWork() {
    return this.currentLevelQueue.length > 0;
  }

  advanceLevel() {
    this.currentLevelQueue = [...this.nextLevelQueue];
    this.nextLevelQueue = [];
  }

  registerOrLink(linkInfo, parentContext) {
    if (!linkInfo || !linkInfo.url || !parentContext) {
      return false;
    }

    const childUrl = linkInfo.url;

    if (!this.urlToContextMap.has(childUrl)) {
      const childContext = new PageContext(
        childUrl,
        linkInfo.title,
        parentContext.depth + 1,
        parentContext
      );
      childContext.isNestedUnderParent = true;
      childContext.setDisplayTitle(linkInfo.title);
      if (linkInfo.section) childContext.setSection(linkInfo.section);
      if (linkInfo.subsection) childContext.setSubsection(linkInfo.subsection);

      parentContext.addChild(childContext);
      this.urlToContextMap.set(childUrl, childContext);
      this._enqueueForNextLevel(childUrl);
      return true;
    }

    const existingContext = this.urlToContextMap.get(childUrl);
    parentContext.addChild(existingContext);
    return false;
  }

  _enqueueForNextLevel(url) {
    if (!url) return;
    if (!this.nextLevelQueue.includes(url)) {
      this.nextLevelQueue.push(url);
    }
  }

  getAllContexts() {
    if (!this.rootContext) {
      return [];
    }
    const result = [];
    const queue = [this.rootContext];
    const visited = new Set();

    while (queue.length > 0) {
      const ctx = queue.shift();
      if (visited.has(ctx)) {
        continue;
      }
      visited.add(ctx);
      result.push(ctx);
      ctx.children.forEach(child => queue.push(child));
    }

    return result;
  }
}

StateManager.instance = null;

module.exports = StateManager;
