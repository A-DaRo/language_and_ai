# Quick Start Guide

This guide provides instructions for users new to Node.js or command-line interfaces. Follow these steps to configure and execute the Notion Scraper.

## 1. Prerequisites

**Node.js** must be installed on the host machine.

1.  Navigate to the [Node.js official website](https://nodejs.org/).
2.  Download the **LTS (Long Term Support)** version appropriate for the operating system (Windows, macOS, or Linux).
3.  Execute the installer and follow the provided instructions.
4.  To verify the installation, open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and execute:
    ```bash
    node -v
    ```
    If a version number is displayed (e.g., `v18.16.0`), the installation is successful.

## 2. Installation

1.  **Download the Source Code**: Download the project directory to the local machine.
2.  **Open Terminal**:
    *   **Windows**: Right-click the project folder and select "Open in Terminal" (or open Command Prompt and navigate to the folder using `cd`).
    *   **macOS/Linux**: Open Terminal and navigate to the folder using `cd /path/to/folder`.
3.  **Install Dependencies**:
    Execute the following command to install all necessary libraries:
    ```bash
    npm install
    ```
    *Note: This process may require several minutes. A progress indicator will be displayed.*

## 3. Configuration

The scraper requires the target Notion page URL to be configured.

1.  Open the file `src/core/Config.js` in a text editor (e.g., Notepad, TextEdit, or VS Code).
2.  Locate the `constructor` section:

    ```javascript
    class Config {
      constructor() {
        // Main configuration
        this.NOTION_PAGE_URL = 'https://your-notion-site-url-here';
        this.OUTPUT_DIR = './downloaded_content';
        this.MAX_RECURSION_DEPTH = 100;
        // ...
    ```

3.  **Set the URL**: Update `this.NOTION_PAGE_URL` with the URL of the target Notion page.
    *   *Important*: The Notion page must be **publicly accessible**.
4.  **Set Output Directory**: Update `this.OUTPUT_DIR` to specify the destination folder (default is `./`, representing the current directory).
5.  **Save the file**.

## 4. Execution

1.  In the terminal, execute:
    ```bash
    npm start
    ```
    *(Alternatively: `node main-cluster.js`)*

2.  **Monitor Progress**:
    *   The system will initiate a "discovery" phase to map the site structure.
    *   A hierarchical tree structure of discovered pages will be displayed.

3.  **Confirm Download**:
    *   The system will pause and request confirmation: `Do you want to proceed with downloading X pages? (y/n)`
    *   Enter `y` and press **Enter**.

4.  **Await Completion**:
    *   Progress indicators for each worker process will be displayed.
    *   Wait for the "Scraping Complete" message.

## 5. Result Verification

1.  Navigate to the installation directory (or the custom `OUTPUT_DIR`).
2.  Locate the folder named after the Notion page (e.g., `JBC090_Language_AI`).
3.  Open the `index.html` file within that folder.
4.  **Open `index.html`** in a web browser.
5.  The entire site is now available for offline browsing.

## Troubleshooting

### "npm is not recognized"
*   **Cause**: Node.js is not installed correctly or the terminal session requires a restart.
*   **Resolution**: Reinstall Node.js and restart the computer.

### "TimeoutError" or "Navigation failed"
*   **Cause**: Network latency or slow response from Notion.
*   **Resolution**: Open `src/core/Config.js` and increase `this.TIMEOUT_PAGE_LOAD` to `120000` (2 minutes).

### "Target closed" or "Browser disconnected"
*   **Cause**: A worker process terminated unexpectedly (typically due to memory constraints).
*   **Resolution**: The system is designed to recover automatically. If this issue persists, close other applications to release system memory.

### "EPERM" or "Permission denied"
*   **Cause**: The scraper lacks write permissions for the target directory.
*   **Resolution**: Execute the terminal as Administrator, or change `OUTPUT_DIR` to a user-writable location (e.g., `C:\Users\YourName\Desktop\NotionDump`).

### Missing Images
*   **Cause**: Certain images may be hosted on external domains that restrict scraping.
*   **Resolution**: Consult the logs in the `logs/` directory for specific error messages.

## Advanced Usage

For advanced configuration and architectural details, refer to:
*   `Docs/README.md`: Comprehensive documentation.
*   `Docs/ARCHITECTURE.md`: Internal system architecture.
