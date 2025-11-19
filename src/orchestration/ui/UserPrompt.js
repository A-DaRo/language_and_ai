const readline = require('readline');

/**
 * Handles interactive CLI prompts for user decisions during discovery.
 * 
 * @classdesc Manages user interaction during the planning phase, allowing users to:
 * - Proceed with scraping the discovered structure
 * - Abort the operation
 * - Request deeper discovery (increase depth by 1)
 * 
 * This class abstracts the readline interaction and normalizes user input into
 * standardized decision values.
 * 
 * @see NotionScraper
 */
class UserPrompt {
  /**
   * Create a new UserPrompt instance.
   */
  constructor() {
    // No dependencies needed
  }

  /**
   * @summary Prompt user for decision on the discovered plan.
   * 
   * Displays an interactive prompt asking the user to confirm, reject, or request
   * deeper exploration of the site structure. Normalizes various input formats
   * (y/yes, n/no, d/deeper) into standardized responses.
   * 
   * @async
   * @returns {Promise<'yes'|'no'|'deeper'>} The user's decision.
   * 
   * @example
   * const prompt = new UserPrompt();
   * const decision = await prompt.promptForPlanDecision();
   * if (decision === 'yes') {
   *   // Proceed with scraping
   * } else if (decision === 'no') {
   *   // Abort
   * } else if (decision === 'deeper') {
   *   // Increase depth and re-scan
   * }
   */
  async promptForPlanDecision() {
    const rl = readline.createInterface({ 
      input: process.stdin, 
      output: process.stdout 
    });
    
    const promptText = '[PROMPT] Do you want to proceed with scraping this structure?\n' +
                      '> Enter (Y)es to continue, (n)o to abort, or (d)eeper to expand the search by one level: ';
    
    const answer = await new Promise(resolve => {
      rl.question(promptText, response => {
        rl.close();
        resolve(response);
      });
    });

    const normalized = (answer || '').trim().toLowerCase();
    
    if (normalized === 'n' || normalized === 'no') {
      return 'no';
    }
    
    if (normalized === 'd' || normalized === 'deeper') {
      return 'deeper';
    }
    
    // Default to 'yes' for any other input (including Enter/empty string)
    return 'yes';
  }
}

module.exports = UserPrompt;
