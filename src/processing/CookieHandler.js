/**
 * @classdesc Handles cookie consent banners on Notion pages.
 * 
 * Implements intelligent cookie consent handling with retry logic and deduplication.
 * Uses a WeakSet to track processed pages, preventing redundant consent operations
 * while allowing reconfiguration if needed.
 * 
 * Features:
 * - Automatic banner detection and dismissal
 * - Configurable retry logic with exponential backoff
 * - Per-page tracking to avoid duplicate processing
 * - Safe handling of closed/invalid page contexts
 * - Graceful degradation on failures
 * 
 * @see PageProcessor#scrapePage
 * @see Config
 */
class CookieHandler {
  /**
   * @param {Config} config - Configuration object for retry settings and behavior flags.
   * @param {Logger} logger - Logger instance for consent handling progress.
   */
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
    this.processedPages = new WeakSet();
  }

  /**
   * @summary Backwards-compatibility shim for existing callers.
   * 
   * @description Delegates to ensureConsent() to maintain API compatibility with older code.
   * 
   * @param {Page} page - Puppeteer page instance.
   * @param {string} [label=''] - Optional label for logging.
   * @returns {Promise<boolean>} True if banner was found and handled.
   * 
   * @deprecated Use ensureConsent() directly.
   * @see ensureConsent
   */
  async handle(page, label = '') {
    return this.ensureConsent(page, label);
  }

  /**
   * @summary Public entry point that safely handles the cookie banner once per page context.
   * 
   * @description Manages the complete cookie consent workflow:
   * 1. Checks if page is usable (not closed)
   * 2. Verifies if consent already processed (unless COOKIE_HANDLE_ALL_PAGES=true)
   * 3. Attempts banner dismissal with configurable retries
   * 4. Tracks successfully processed pages to prevent duplicates
   * 
   * Uses WeakSet for page tracking, automatically garbage collected when pages close.
   * 
   * @param {Page} page - Puppeteer page instance.
   * @param {string} [label=''] - Optional label for logging (defaults to page URL).
   * @returns {Promise<boolean>} True if banner was found and handled, false otherwise.
   * 
   * @see Config#COOKIE_HANDLE_ALL_PAGES
   * @see Config#COOKIE_MAX_RETRIES
   */
  async ensureConsent(page, label = '') {
    if (!this._isPageUsable(page)) {
      this.logger.warn('COOKIE', 'Cannot handle cookies because the page is closed.');
      return false;
    }

    if (this.processedPages.has(page)) {
      if (!this.config.COOKIE_HANDLE_ALL_PAGES) {
        this.logger.debug('COOKIE', 'Cookie consent already handled for this page context.');
        return false;
      }
      this.logger.debug('COOKIE', 'Re-running cookie consent per configuration.');
    }

    const targetLabel = label || (await this._safePageUrl(page)) || 'page';
    this.logger.info('COOKIE', `Ensuring cookie consent for ${targetLabel}...`);

    let attempt = 0;
    const maxAttempts = Math.max(0, this.config.COOKIE_MAX_RETRIES || 0) + 1;

    while (attempt < maxAttempts) {
      if (!this._isPageUsable(page)) {
        this.logger.warn('COOKIE', 'Page became unavailable while handling cookies.');
        break;
      }

      const result = await this._attemptHandle(page);
      if (result.completed) {
        this.processedPages.add(page);
        return result.bannerFound;
      }

      attempt += 1;
      if (attempt < maxAttempts && result.shouldRetry) {
        const delay = this.config.COOKIE_RETRY_DELAY * attempt;
        this.logger.warn('COOKIE', `Retrying cookie handling in ${delay}ms (attempt ${attempt + 1}/${maxAttempts}).`);
        await this._delay(delay);
      } else {
        break;
      }
    }

    this.processedPages.add(page);
    return false;
  }

  async _attemptHandle(page) {
    try {
      await page.waitForSelector(this.config.SELECTORS.cookieBanner, {
        timeout: this.config.TIMEOUT_COOKIE_BANNER
      });
      this.logger.info('COOKIE', 'Cookie banner detected.');
    } catch (error) {
      this.logger.debug('COOKIE', 'No cookie banner detected within timeout.');
      return { completed: true, bannerFound: false, shouldRetry: false };
    }

    const rejectButtonClicked = await this._clickRejectButton(page);

    if (rejectButtonClicked) {
      this.logger.info('COOKIE', 'Clicked "Reject all" button.');
      await this._handleConfirmationDialog(page);
      return { completed: true, bannerFound: true, shouldRetry: false };
    }

    this.logger.warn('COOKIE', 'Reject button missing, scheduling retry.');
    return { completed: false, bannerFound: true, shouldRetry: true };
  }

  /**
   * Click the "Reject all" button
   */
  async _clickRejectButton(page) {
    if (!this._isPageUsable(page)) {
      return false;
    }

    try {
      return await page.evaluate(() => {
        const buttons = Array.from(document.querySelectorAll('div[role="button"]'));
        const rejectButton = buttons.find(btn => btn.textContent && btn.textContent.includes('Reject'));
        if (rejectButton) {
          rejectButton.click();
          return true;
        }
        return false;
      });
    } catch (error) {
      this.logger.error('COOKIE', 'Error while attempting to click the reject button', error);
      return false;
    }
  }

  /**
   * Handle the confirmation dialog that appears after rejecting cookies
   */
  async _handleConfirmationDialog(page) {
    if (!this._isPageUsable(page)) {
      return;
    }

    try {
      await this._delay(500);
      this.logger.info('COOKIE', 'Waiting for confirmation dialog...');

      await page.waitForSelector(this.config.SELECTORS.confirmDialog, {
        timeout: this.config.TIMEOUT_DIALOG
      });
      this.logger.info('COOKIE', 'Confirmation dialog detected.');

      const okButtonClicked = await page.evaluate(() => {
        const dialog = document.querySelector('div[role="dialog"][aria-modal="true"]');
        if (dialog) {
          const buttons = Array.from(dialog.querySelectorAll('div[role="button"]'));
          const okButton = buttons.find(btn => btn.textContent && btn.textContent.trim().toLowerCase() === 'ok');
          if (okButton) {
            okButton.click();
            return true;
          }
        }
        return false;
      });

      if (okButtonClicked) {
        this.logger.info('COOKIE', 'Clicked "OK" button. Waiting for stability...');

        await Promise.race([
          page.waitForNavigation({
            waitUntil: 'networkidle0',
            timeout: this.config.TIMEOUT_NAVIGATION
          }).catch(() => null),
          this._delay(this.config.WAIT_AFTER_COOKIE)
        ]);

        this.logger.success('COOKIE', 'Confirmation dialog handled successfully.');
      } else {
        this.logger.warn('COOKIE', 'Could not find "OK" button in dialog.');
      }
    } catch (error) {
      this.logger.error('COOKIE', 'Error handling confirmation dialog', error);
    }
  }

  _isPageUsable(page) {
    return page && typeof page.isClosed === 'function' && !page.isClosed();
  }

  async _safePageUrl(page) {
    try {
      return await page.url();
    } catch (error) {
      return null;
    }
  }

  async _delay(duration) {
    return new Promise(resolve => setTimeout(resolve, duration));
  }
}

module.exports = CookieHandler;
