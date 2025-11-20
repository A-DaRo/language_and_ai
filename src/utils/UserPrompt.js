const readline = require('readline');

/**
 * Terminal interaction utility for user confirmation prompts.
 * Provides readline-based interactive prompts with timeout handling,
 * input validation, and graceful SIGINT (Ctrl+C) handling.
 * 
 * @class UserPrompt
 */
class UserPrompt {
  /**
   * @param {Object} [mockReadline] - Optional mock readline interface for testing
   */
  constructor(mockReadline = null) {
    this.rl = mockReadline || null;
    this.sigintHandler = null;
    this.isMocked = !!mockReadline;
  }

  /**
   * Create readline interface (lazy initialization)
   * @private
   */
  _ensureReadline() {
    if (!this.rl) {
      if (this.isMocked) {
        throw new Error('Mock readline was not properly initialized');
      }
      this.rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
      });
    }
  }

  /**
   * Check if running in interactive terminal environment
   * @returns {boolean} True if stdin is a TTY
   * @private
   */
  _isInteractive() {
    return process.stdin.isTTY;
  }

  /**
   * Prompt user with yes/no question
   * 
   * @param {string} question - Question to display
   * @param {boolean|null} defaultAnswer - Default answer (null = no default)
   * @param {number} timeout - Timeout in milliseconds (default 60000)
   * @returns {Promise<boolean>} True for yes, false for no/timeout/abort
   * 
   * @example
   * const prompt = new UserPrompt();
   * const proceed = await prompt.promptYesNo('Continue?', null, 60000);
   * prompt.close();
   */
  async promptYesNo(question, defaultAnswer = null, timeout = 60000) {
    // Non-interactive environment check
    if (!this._isInteractive()) {
      console.warn('Running in non-interactive mode. Using default behavior.');
      return defaultAnswer !== null ? defaultAnswer : true;
    }

    this._ensureReadline();

    const defaultText = defaultAnswer === true ? ' (Y/n)' : 
                        defaultAnswer === false ? ' (y/N)' : 
                        ' (y/n)';

    let attempts = 0;
    const maxAttempts = 3;

    while (attempts < maxAttempts) {
      const answer = await this._promptWithTimeout(
        `${question}${defaultText}: `,
        timeout
      );

      // Handle timeout or SIGINT
      if (answer === null) {
        return false;
      }

      // Normalize input
      const normalized = answer.trim().toLowerCase();

      // Empty input uses default
      if (normalized === '') {
        if (defaultAnswer !== null) {
          return defaultAnswer;
        }
        console.log('Please enter y or n.');
        attempts++;
        continue;
      }

      // Validate input
      if (normalized === 'y' || normalized === 'yes') {
        return true;
      }
      if (normalized === 'n' || normalized === 'no') {
        return false;
      }

      // Invalid input
      console.log('Invalid input. Please enter y or n.');
      attempts++;
    }

    // Max attempts reached
    console.warn('Too many invalid attempts. Aborting.');
    return false;
  }

  /**
   * Specialized prompt for download confirmation with statistics
   * 
   * @param {Object} stats - Discovery statistics
   * @param {number} stats.totalPages - Total pages discovered
   * @param {number} stats.maxDepth - Maximum depth reached
   * @param {number} [stats.conflicts] - Number of path conflicts
   * @returns {Promise<boolean>} True to proceed, false to abort
   * 
   * @example
   * const stats = { totalPages: 42, maxDepth: 3, conflicts: 2 };
   * const proceed = await prompt.promptConfirmDownload(stats);
   */
  async promptConfirmDownload(stats) {
    const message = `\nProceed with downloading ${stats.totalPages} page${stats.totalPages !== 1 ? 's' : ''}?`;
    return this.promptYesNo(message, null, 60000);
  }

  /**
   * Internal method to prompt with timeout and SIGINT handling
   * 
   * @param {string} query - Question to display
   * @param {number} timeout - Timeout in milliseconds
   * @returns {Promise<string|null>} User input or null on timeout/SIGINT
   * @private
   */
  _promptWithTimeout(query, timeout) {
    return new Promise((resolve) => {
      let timeoutId;
      let resolved = false;

      // Setup SIGINT handler (Ctrl+C)
      this.sigintHandler = () => {
        if (!resolved) {
          resolved = true;
          clearTimeout(timeoutId);
          console.log('\n^C (Aborted by user)');
          resolve(null);
        }
      };
      process.once('SIGINT', this.sigintHandler);

      // Setup timeout
      timeoutId = setTimeout(() => {
        if (!resolved) {
          resolved = true;
          process.removeListener('SIGINT', this.sigintHandler);
          console.log('\nTimeout reached. Aborting.');
          resolve(null);
        }
      }, timeout);

      // Ask question
      this.rl.question(query, (answer) => {
        if (!resolved) {
          resolved = true;
          clearTimeout(timeoutId);
          process.removeListener('SIGINT', this.sigintHandler);
          resolve(answer);
        }
      });
    });
  }

  /**
   * Close readline interface and cleanup resources
   * Should be called when done with prompts to prevent hanging process
   * 
   * @example
   * const prompt = new UserPrompt();
   * await prompt.promptYesNo('Continue?');
   * prompt.close(); // Important: cleanup
   */
  close() {
    if (this.rl) {
      this.rl.close();
      this.rl = null;
    }
    if (this.sigintHandler) {
      process.removeListener('SIGINT', this.sigintHandler);
      this.sigintHandler = null;
    }
  }
}

module.exports = UserPrompt;
