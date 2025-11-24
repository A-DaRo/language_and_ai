/**
 * @fileoverview Phase 3: User Confirmation - Display tree and prompt for confirmation
 * @module orchestration/phases/UserConfirmationPhase
 * @description Allows user to review discovered structure before download.
 */

const UserPrompt = require('../../utils/UserPrompt');
const PhaseStrategy = require('./PhaseStrategy');

/**
 * @class UserConfirmationPhase
 * @extends PhaseStrategy
 * @classdesc Displays site hierarchy and prompts user for confirmation.
 */
class UserConfirmationPhase extends PhaseStrategy {
  /**
   * @async
   * @method execute
   * @summary Displays the site tree and prompts for confirmation
   * @description
   * 1. Generates visual representation of discovered hierarchy
   * 2. Displays summary statistics (total pages, depth)
   * 3. Prompts user (Y/N) via UserPrompt
   * 4. Aborts if declined
   * @returns {Promise<boolean>} True to proceed, false to abort
   */
  async execute() {
    this.orchestrator.eventBus.emit('PHASE:STOPPING_DASHBOARD', {});
    await new Promise(resolve => setTimeout(resolve, 100));

    this.logger.separator('Phase 3: User Confirmation');

    const stats = this.queueManager.getStatistics();
    if (stats.discovered === 0) {
      this.logger.warn('USER_CONFIRMATION', 'No pages discovered. Aborting.');
      return false;
    }

    // Display tree
    this.logger.info('USER_CONFIRMATION', 'Discovered Site Structure:');
    const allContexts = this.queueManager.getAllContexts();
    const rootContext = allContexts.find(ctx => ctx.depth === 0);
    if (rootContext) {
      this._displayPageTree(rootContext);
    }

    // Display summary
    this.logger.info('USER_CONFIRMATION', '\nDiscovery Summary:');
    this.logger.info('USER_CONFIRMATION', `  Total Pages: ${stats.discovered}`);
    this.logger.info('USER_CONFIRMATION', `  Maximum Depth: ${this.queueManager.getMaxDepth()}`);

    // Prompt user
    const prompt = new UserPrompt();
    const proceed = await prompt.promptConfirmDownload({
      totalPages: stats.discovered,
      maxDepth: this.queueManager.getMaxDepth()
    });
    prompt.close();

    if (proceed) {
      this.logger.success('USER_CONFIRMATION', 'User confirmed. Proceeding...');
    } else {
      this.logger.warn('USER_CONFIRMATION', 'User declined. Aborting.');
    }

    return proceed;
  }

  /**
   * @private
   * @param {PageContext} rootContext - Root page context
   */
  _displayPageTree(rootContext) {
    this.logger.separator('Page Tree');
    console.log('.');

    const titleRegistry = this.queueManager.getTitleRegistry();
    const rootLabel = titleRegistry[rootContext.id] || rootContext.title || '(root)';
    const maxBfsDepth = this.queueManager.getMaxDepth();
    
    console.log(`└─ ${rootLabel}`);

    rootContext.children.forEach((child, index) => {
      const isLast = index === rootContext.children.length - 1;
      this._printTreeNode(
        child,
        '   ',
        isLast,
        titleRegistry,
        new Set([rootContext.id]),
        maxBfsDepth
      );
    });

    this.logger.separator();
  }

  /**
   * @private
   * @param {PageContext} context - Page context
   * @param {string} prefix - Line prefix
   * @param {boolean} isLast - Is this the last child?
   * @param {Object} titleRegistry - ID-to-title map
   * @param {Set<string>} pathVisited - Visited nodes in current path (for cycle detection)
   * @param {number} maxBfsDepth - Maximum depth of BFS expansion (limit recursion to maxBfsDepth+1)
   */
  _printTreeNode(context, prefix, isLast, titleRegistry, pathVisited = new Set(), maxBfsDepth = Infinity) {
    const connector = isLast ? '└─ ' : '├─ ';
    const title = titleRegistry[context.id] || context.title || 'Untitled';

    // Detect cycles within the path (visited in current recursion)
    if (pathVisited.has(context.id)) {
      console.log(`${prefix}${connector}${title} ↺ (Cycle)`);
      return;
    }

    // Filter to only show discovered children (have titles in registry)
    const exploredChildren = context.children.filter(child => titleRegistry[child.id]);
    const internalRefs = context.children.length - exploredChildren.length;
    const label = internalRefs > 0 ? `${title} [${internalRefs} internal ref${internalRefs > 1 ? 's' : ''}]` : title;

    console.log(`${prefix}${connector}${label}`);

    // Stop recursion at BFS depth + 1 to limit tree representation
    // This shows the full BFS expansion plus edges creating cycles at leaves
    if (context.depth >= maxBfsDepth) {
      // We're at or past the BFS leaf level - show children as cycles only
      for (const child of exploredChildren) {
        if (pathVisited.has(child.id)) {
          // Only show explicitly detected cycles
          const connector2 = child === exploredChildren[exploredChildren.length - 1] ? '└─ ' : '├─ ';
          const childTitle = titleRegistry[child.id] || child.title || 'Untitled';
          const childPrefix = prefix + (isLast ? '   ' : '│  ');
          console.log(`${childPrefix}${connector2}${childTitle} ↺ (Cycle)`);
        }
        // Don't recurse deeper
      }
      return;
    }

    // Recurse into children (we're within BFS expansion depth)
    const childPrefix = prefix + (isLast ? '   ' : '│  ');
    const nextVisited = new Set(pathVisited);
    nextVisited.add(context.id);
    
    exploredChildren.forEach((child, index) => {
      const childIsLast = index === exploredChildren.length - 1;
      
      // Only recurse if this is a tree edge (child's parent is current context)
      if (child.parentContext === context) {
        this._printTreeNode(
          child,
          childPrefix,
          childIsLast,
          titleRegistry,
          nextVisited,
          maxBfsDepth
        );
      } else {
        // Non-tree edge (Cycle/Cross-link) - Print leaf and stop
        const connector = childIsLast ? '└─ ' : '├─ ';
        const title = titleRegistry[child.id] || child.title || 'Untitled';
        console.log(`${childPrefix}${connector}${title} ↺ (Cycle)`);
      }
    });
  }
}

module.exports = UserConfirmationPhase;
