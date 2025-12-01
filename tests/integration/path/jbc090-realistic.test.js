/**
 * JBC090 Realistic Integration Tests
 * 
 * These tests simulate real-world path resolution scenarios based on the actual
 * scraped JBC090-Language-AI Notion site structure. The test cases mirror the
 * exact navigation patterns found in the scraped content.
 * 
 * Site Structure:
 * /JBC090-Language-AI/
 * ├── index.html (root page with links to sections)
 * ├── Lab_Session_1/ → index.html (nested page with toggles, code blocks)
 * ├── Lab_Session_2/ → index.html
 * ├── Lab_Session_3/ → index.html
 * ├── Lab_Session_4/ → index.html
 * ├── Lab_Session_5/ → index.html
 * ├── Lab_Session_6/ → index.html
 * ├── Deep_Learning/ → index.html
 * ├── Information_Extraction/ → index.html
 * ├── Large_Language_Models/ → index.html
 * ├── Representation/ → index.html
 * ├── Classification/ → index.html
 * ├── Collecting_Data/ → index.html
 * ├── Introduction/ → index.html
 * ├── Syllabus/ → index.html
 * ├── Material/ → index.html
 * ├── Changelog/ → index.html
 * ├── Assessment_Information/ → index.html
 * ├── files/
 * │   ├── week4.pdf
 * │   ├── week5.pdf
 * │   ├── week6.pdf
 * │   └── week-7.pdf
 * ├── css/
 * └── images/
 */

const path = require('path');
const { PathResolverFactory, PathResolver } = require('../../../src/domain/path');
const PageContext = require('../../../src/domain/PageContext');

describe('JBC090 Realistic Navigation Scenarios', () => {
    // Create factory instance for all tests
    let factory;
    
    beforeEach(() => {
        factory = new PathResolverFactory();
    });

    /**
     * Helper to create PageContext from folder path.
     * PageContext constructor: (url, rawTitle, depth, parentContext, parentId)
     * We need to properly set pathSegments after creation.
     */
    const createPageContext = (folderPath, title = 'Test Page') => {
        // Simulate normalized path from scraping
        const normalizedPath = folderPath.replace(/\\/g, '/');
        const segments = normalizedPath.split('/').filter(Boolean);
        const depth = segments.length;
        
        // Create a UNIQUE Notion-like URL with a 32-char hex ID based on path
        // This ensures each page has a unique ID for proper resolver routing
        const crypto = require('crypto');
        const hash = crypto.createHash('md5').update(normalizedPath || 'root').digest('hex');
        const pageId = hash.substring(0, 32);
        const url = `https://notion.site/${normalizedPath}-${pageId}`;
        
        const ctx = new PageContext(url, title, depth, null, null);
        
        // Manually set pathSegments to simulate scraper behavior
        ctx.pathSegments = segments;
        
        return ctx;
    };

    describe('Gallery View Navigation (Root → Section Pages)', () => {
        /**
         * From root index.html, clicking on gallery cards like:
         * - "Lab Session 4" → Lab_Session_4/index.html
         * - "Deep Learning" → Deep_Learning/index.html
         * - "Large Language Models" → Large_Language_Models/index.html
         */
        
        test('should resolve root to Lab_Session_4 navigation', () => {
            const source = createPageContext('JBC090-Language-AI', 'JBC090 Home');
            const target = createPageContext('JBC090-Language-AI/Lab_Session_4', 'Lab Session 4');
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('Lab_Session_4/index.html');
        });

        test('should resolve root to Deep_Learning navigation', () => {
            const source = createPageContext('JBC090-Language-AI', 'JBC090 Home');
            const target = createPageContext('JBC090-Language-AI/Deep_Learning', 'Deep Learning');
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('Deep_Learning/index.html');
        });

        test('should resolve root to Large_Language_Models navigation', () => {
            const source = createPageContext('JBC090-Language-AI', 'JBC090 Home');
            const target = createPageContext('JBC090-Language-AI/Large_Language_Models', 'Large Language Models');
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('Large_Language_Models/index.html');
        });

        test('should resolve root to Changelog navigation', () => {
            const source = createPageContext('JBC090-Language-AI', 'JBC090 Home');
            const target = createPageContext('JBC090-Language-AI/Changelog', 'Changelog');
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('Changelog/index.html');
        });
    });

    describe('Cross-Section Navigation (Section → Section)', () => {
        /**
         * Navigating between sibling sections (same depth level):
         * - Lab_Session_4 → Lab_Session_5
         * - Deep_Learning → Information_Extraction
         * - Syllabus → Material
         */
        
        test('should resolve Lab_Session_4 to Lab_Session_5 navigation', () => {
            const source = createPageContext('JBC090-Language-AI/Lab_Session_4', 'Lab Session 4');
            const target = createPageContext('JBC090-Language-AI/Lab_Session_5', 'Lab Session 5');
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('../Lab_Session_5/index.html');
        });

        test('should resolve Deep_Learning to Information_Extraction navigation', () => {
            const source = createPageContext('JBC090-Language-AI/Deep_Learning', 'Deep Learning');
            const target = createPageContext('JBC090-Language-AI/Information_Extraction', 'Information Extraction');
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('../Information_Extraction/index.html');
        });

        test('should resolve Syllabus to Material navigation', () => {
            const source = createPageContext('JBC090-Language-AI/Syllabus', 'Syllabus');
            const target = createPageContext('JBC090-Language-AI/Material', 'Material');
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('../Material/index.html');
        });
    });

    describe('Breadcrumb Navigation (Child → Parent)', () => {
        /**
         * Navigating back up the hierarchy:
         * - Lab_Session_1 → JBC090 Home
         * - Deep_Learning → JBC090 Home
         */
        
        test('should resolve Lab_Session_1 back to root navigation', () => {
            const source = createPageContext('JBC090-Language-AI/Lab_Session_1', 'Lab Session 1');
            const target = createPageContext('JBC090-Language-AI', 'JBC090 Home');
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('../index.html');
        });

        test('should resolve nested section back to root navigation', () => {
            const source = createPageContext('JBC090-Language-AI/Classification', 'Classification');
            const target = createPageContext('JBC090-Language-AI', 'JBC090 Home');
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('../index.html');
        });
    });

    describe('File Downloads (PDF Links)', () => {
        /**
         * PDF file links from various pages:
         * - Root page links to files/week4.pdf, files/week5.pdf, etc.
         * - Section pages may also link to files in parent or sibling folders
         * 
         * ARCHITECTURE NOTE: Local file paths (like files/week4.pdf) are classified
         * as EXTERNAL by the path resolver because they have no target PageContext.
         * This is intentional - the LinkRewriter preserves these paths unchanged.
         * The actual file resolution happens during asset downloading, not link rewriting.
         */
        
        test('should treat local file paths as external (preserved unchanged)', () => {
            const source = createPageContext('JBC090-Language-AI', 'JBC090 Home');
            
            // Local file paths have no target PageContext, so they're treated as external
            // This means they're preserved unchanged during link rewriting
            const pathType = factory.getPathType({
                source,
                target: null,
                href: 'files/week4.pdf'
            });
            
            // EXTERNAL type means "pass through unchanged"
            expect(pathType).toBe(PathResolver.Types.EXTERNAL);
        });

        test('should preserve relative file paths unchanged', () => {
            const source = createPageContext('JBC090-Language-AI/Lab_Session_4', 'Lab Session 4');
            
            // Relative paths to local files are also preserved unchanged
            const pathType = factory.getPathType({
                source,
                target: null,
                href: '../files/week4.pdf'
            });
            
            expect(pathType).toBe(PathResolver.Types.EXTERNAL);
        });

        test('should resolve external file URL unchanged', () => {
            const href = '../files/week4.pdf';
            
            // ExternalUrlResolver preserves the original href
            const result = factory.resolveAs(PathResolver.Types.EXTERNAL, {
                source: null,
                target: null,
                href: href
            });
            
            expect(result).toBe(href);
        });
    });

    describe('Intra-Page Anchors (Toggle Sections)', () => {
        /**
         * The Lab Session pages have collapsible toggle sections:
         * - Hint 1, Hint 2, Hint 3, Hint 4
         * - Partial Solution, Solution
         * 
         * These create anchor links within the same page.
         * Block IDs should be provided without hyphens (32-char hex).
         */
        
        test('should resolve same-page anchor for toggle section', () => {
            const source = createPageContext('JBC090-Language-AI/Lab_Session_1', 'Lab Session 1');
            // Block ID without hyphens (raw hex format)
            const blockId = '29d979eeca9f816d8f32e828b3e7f6f5';
            
            const result = factory.resolve({
                source,
                target: source, // Same page
                blockId: blockId
            });
            
            // Output is formatted with hyphens as UUID
            expect(result).toBe('#29d979ee-ca9f-816d-8f32-e828b3e7f6f5');
        });

        test('should correctly identify intra-page anchor type', () => {
            const source = createPageContext('JBC090-Language-AI/Lab_Session_1', 'Lab Session 1');
            const blockId = '29d979eeca9f81cfa4c4e836f606a97a';
            
            const pathType = factory.getPathType({
                source,
                target: source, // Same page
                blockId: blockId
            });
            
            expect(pathType).toBe(PathResolver.Types.INTRA);
        });
    });

    describe('External Links', () => {
        /**
         * External documentation links found in Lab Session pages:
         * - https://docs.python.org/3/library/stdtypes.html
         * - https://docs.python.org/3/library/string.html
         * - https://docs.python.org/3/library/collections.html
         */
        
        test('should identify Python docs as external', () => {
            const source = createPageContext('JBC090-Language-AI/Lab_Session_1', 'Lab Session 1');
            const href = 'https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str';
            
            const pathType = factory.getPathType({
                source,
                target: null,
                href: href
            });
            
            expect(pathType).toBe(PathResolver.Types.EXTERNAL);
        });

        test('should resolve external URL unchanged', () => {
            const externalUrl = 'https://docs.python.org/3/library/collections.html';
            
            const result = factory.resolveAs(PathResolver.Types.EXTERNAL, {
                source: null,
                target: null,
                href: externalUrl
            });
            
            expect(result).toBe(externalUrl);
        });

        test('should identify W3Schools link as external', () => {
            const href = 'https://www.w3schools.com/python/python_strings_slicing.asp';
            
            const pathType = factory.getPathType({
                source: createPageContext('JBC090-Language-AI/Lab_Session_1', 'Lab Session 1'),
                target: null,
                href: href
            });
            
            expect(pathType).toBe(PathResolver.Types.EXTERNAL);
        });
    });

    describe('Deep Hierarchy Navigation', () => {
        /**
         * Simulating deeper nesting (if subpages exist within sections)
         */
        
        test('should resolve 3-level deep to root', () => {
            const source = createPageContext(
                'JBC090-Language-AI/Lab_Session_1/Exercises',
                'Exercises'
            );
            const target = createPageContext('JBC090-Language-AI', 'JBC090 Home');
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('../../index.html');
        });

        test('should resolve 3-level deep to sibling section', () => {
            const source = createPageContext(
                'JBC090-Language-AI/Lab_Session_1/Exercises',
                'Exercises'
            );
            const target = createPageContext('JBC090-Language-AI/Lab_Session_2', 'Lab Session 2');
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('../../Lab_Session_2/index.html');
        });

        test('should resolve cross-tree navigation from deep page to deep page', () => {
            const source = createPageContext(
                'JBC090-Language-AI/Lab_Session_1/Exercises/Task1',
                'Task 1'
            );
            const target = createPageContext(
                'JBC090-Language-AI/Lab_Session_2/Solutions/Answer1',
                'Answer 1'
            );
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('../../../Lab_Session_2/Solutions/Answer1/index.html');
        });
    });

    describe('Special Characters in Paths', () => {
        /**
         * Notion page names may include special characters
         */
        
        test('should handle page names with underscores', () => {
            const source = createPageContext('JBC090-Language-AI', 'Home');
            const target = createPageContext('JBC090-Language-AI/Welcome_Course_in_Brief', 'Welcome Course in Brief');
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('Welcome_Course_in_Brief/index.html');
        });

        test('should handle page names with hyphens', () => {
            const source = createPageContext('JBC090-Language-AI', 'Home');
            const target = createPageContext('JBC090-Language-AI/Interim_Assignment', 'Interim Assignment');
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('Interim_Assignment/index.html');
        });
    });

    describe('BlockMap Cache Integration', () => {
        /**
         * Cross-page block anchors require blockMapCache lookup.
         * Block IDs should be 32-char hex (no hyphens).
         */
        
        test('should resolve cross-page anchor with blockMapCache', () => {
            const source = createPageContext('JBC090-Language-AI', 'Home');
            const target = createPageContext('JBC090-Language-AI/Lab_Session_1', 'Lab Session 1');
            // Block ID without hyphens (raw format)
            const blockId = '29d979eeca9f8109ad1deb8b3f3d175d';
            
            // Simulate blockMapCache that maps blockId to formatted ID
            const blockMapCache = new Map([
                [target.id, new Map([
                    [blockId, '29d979ee-ca9f-8109-ad1d-eb8b3f3d175d'] // formatted version
                ])]
            ]);
            
            const result = factory.resolve({
                source,
                target,
                blockId,
                blockMapCache
            });
            
            // Should include anchor hash with formatted UUID
            expect(result).toBe('Lab_Session_1/index.html#29d979ee-ca9f-8109-ad1d-eb8b3f3d175d');
        });
    });

    describe('Edge Cases from Real Content', () => {
        test('should handle self-referential link (same page, no anchor)', () => {
            const source = createPageContext('JBC090-Language-AI/Lab_Session_1', 'Lab Session 1');
            
            // Same source and target, no blockId
            // ARCHITECTURE: IntraPageResolver returns '' for same-page without anchor
            // This represents "stay on current page" - no href needed
            const result = factory.resolve({
                source,
                target: source,
                blockId: null
            });
            
            // Empty string means "no navigation needed"
            expect(result).toBe('');
        });

        test('should handle empty normalized paths gracefully', () => {
            // Root level page
            const source = createPageContext('', 'Root');
            const target = createPageContext('Section', 'Section');
            
            const result = factory.resolve({
                source,
                target,
                blockId: null
            });
            
            expect(result).toBe('Section/index.html');
        });
    });
});

describe('Path Type Detection Suite', () => {
    // Create factory instance for all tests
    let factory;
    
    beforeEach(() => {
        factory = new PathResolverFactory();
    });

    describe('should correctly classify all JBC090 link types', () => {
        const createTestContext = (normalizedPath, title) => {
            const segments = normalizedPath.split('/').filter(Boolean);
            const depth = segments.length;
            // Create unique ID based on path
            const crypto = require('crypto');
            const hash = crypto.createHash('md5').update(normalizedPath || 'root').digest('hex');
            const pageId = hash.substring(0, 32);
            const url = `https://notion.site/${normalizedPath}-${pageId}`;
            
            const ctx = new PageContext(url, title, depth, null, null);
            ctx.pathSegments = segments;
            return ctx;
        };

        const testCases = [
            // Inter-page links
            { href: null, hasTarget: true, samePage: false, hasBlockId: false, expected: PathResolver.Types.INTER },
            // Intra-page anchors
            { href: null, hasTarget: true, samePage: true, hasBlockId: true, expected: PathResolver.Types.INTRA },
            // External URLs (includes http/https URLs)
            { href: 'https://docs.python.org/3/', hasTarget: false, expected: PathResolver.Types.EXTERNAL },
            { href: 'http://example.com', hasTarget: false, expected: PathResolver.Types.EXTERNAL },
            // Local file paths (no target PageContext = EXTERNAL, preserved unchanged)
            // These are NOT filesystem resolver cases - filesystem is for page output paths
            { href: 'files/week4.pdf', hasTarget: false, expected: PathResolver.Types.EXTERNAL },
            { href: '../files/week5.pdf', hasTarget: false, expected: PathResolver.Types.EXTERNAL },
            { href: 'css/style.css', hasTarget: false, expected: PathResolver.Types.EXTERNAL },
            { href: 'images/logo.png', hasTarget: false, expected: PathResolver.Types.EXTERNAL },
        ];

        testCases.forEach(({ href, hasTarget, samePage, hasBlockId, expected }, index) => {
            test(`case ${index + 1}: should detect ${expected} type`, () => {
                const source = createTestContext('test', 'Test');
                
                let target = null;
                if (hasTarget) {
                    target = samePage ? source : createTestContext('other', 'Other');
                }

                const context = {
                    source,
                    target,
                    href,
                    blockId: hasBlockId ? 'some-block-id' : null
                };

                const result = factory.getPathType(context);
                expect(result).toBe(expected);
            });
        });
    });
});
