# Plan: Centralized ID-to-Title Map Architecture Refactor

This plan refactors the system from scattered title-tracking mechanisms to a centralized **ID-to-Title Map** managed by `GlobalQueueManager`. This enables efficient internal operations using raw IDs while providing instant human-readable title lookups for display, logging, and filesystem operations.

## Steps

### 1. Refactor `ProtocolDefinitions.js` to use Map-based title structure

- Change all typedef structures from `title: string` to `titleRegistry: Object` (serialized Map)
- Update `DiscoverPayload`, `DownloadPayload`, `DiscoveryResult`, `DownloadResult` typedefs
- Add `serializeTitleMap(map)` and `deserializeTitleMap(obj)` utility functions
- Update JSDoc examples to reflect new structure: `{id1: 'Clean Title', id2: 'Another Title'}`

### 2. Add centralized title registry to `GlobalQueueManager.js`

- Add `this.idToTitleMap = new Map()` in constructor alongside existing `allContexts` and `visitedUrls`
- Update `completeDiscovery()` to populate map: `this.idToTitleMap.set(pageId, resolvedTitle || 'Untitled')`
- Expose public getter: `getTitleRegistry()` returns `Object.fromEntries(this.idToTitleMap)` for IPC serialization
- Add `getTitleById(id)` and `setTitle(id, title)` convenience methods
- Update all logging statements to use `idToTitleMap.get(pageId)` instead of `context.displayTitle`

### 3. Simplify `PageContext.js` to single title field

- Remove `originalTitle`, `displayTitle` properties (keep only `title` for sanitized filesystem name)
- Remove `setDisplayTitle()` method (no longer needed with centralized registry)
- Update constructor: accept `rawTitle` parameter, sanitize to `this.title`, use `id` for lookups
- Update `toJSON()`/`fromJSON()` to remove obsolete title fields
- Keep `getRelativePath()` using sanitized `title` for filesystem operations
- Update `toString()` method to note "titles resolved via GlobalQueueManager"

### 4. Update `ClusterOrchestrator.js` to pass title registry to workers

- In discovery phase: after `completeDiscovery()`, serialize registry via `queueManager.getTitleRegistry()`
- In download dispatch: include `titleRegistry` in `DownloadPayload` for worker reference
- Update ASCII tree rendering method `_displayPageTree()` to accept `titleRegistry` parameter
- Refactor tree node display: `const title = titleRegistry[context.id] || context.title || 'Untitled'`
- Update all logging to query registry: `logger.info('TASK', titleRegistry[pageId])`

### 5. Refactor `TaskRunner.js` and `PageProcessor.js` worker-side logic

- In `_executeDiscovery()`: return only `{ pageId, resolvedTitle, links }` (no displayTitle field)
- In `_executeDownload()`: receive `payload.titleRegistry` from master
- Update logging to use titleRegistry lookups: `logger.debug('WORKER', titleRegistry[pageId])`
- Remove title resolution fallback logic (single source of truth is master's registry)
- Update link extraction to continue returning raw IDs as placeholders (master resolves them)

### 6. Update `ConflictResolver.js` to use title registry for logging

- Accept `titleRegistry` parameter in `resolve()` static method
- Replace all `context.displayTitle || context.title` patterns with `titleRegistry[context.id]`
- Keep canonical path calculation using `context.title` (sanitized filesystem name)
- Update duplicate detection logging: show human-readable titles from registry
- Pass registry to `_selectCanonical()` for better logging of selection criteria

## Further Considerations

### 1. No Backward compatibility
Do not maintain `displayTitle`. Hard break all old logic.

### 2. Title update strategy
Titles are immutable. They are directly linked to page IDs and they form a bijective mapping. No updates after initial resolution.

### 3. Worker memory efficiency
The query should be handled efficiently. The one-to-one mapping allows for quick lookups without excessive memory overhead. Allowing to query only raw IDs for all internal operations (requesting only keys of the map), and exposing the full map only when necessary (e.g., for logging or display).

### 4. Testing strategy
Add explicit test for title registry synchronization across discovery → pruning → download phases. This should be done within the tests/ subfolder.
