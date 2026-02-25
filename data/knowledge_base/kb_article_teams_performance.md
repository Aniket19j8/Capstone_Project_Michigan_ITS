# Knowledge Base: Microsoft Teams Performance Issues

## Problem
Users report Teams running slowly, consuming high memory, or causing system lag.

## Root Cause
Teams is an Electron-based application that can consume significant RAM (1-2GB+).
Common triggers: many open chats, large files in channels, background activity.

## Solution
1. **Clear Teams cache** (most effective):
   - Close Teams completely (check system tray)
   - Delete contents of: `%appdata%\Microsoft\Teams\Cache`
   - Also clear: `blob_storage`, `databases`, `GPUCache`, `IndexedDB`, `Local Storage`, `tmp`
   - Restart Teams

2. **Reduce memory usage**:
   - Close unused chats and channels
   - Disable GPU hardware acceleration: Settings → General → Uncheck "Disable GPU hardware acceleration"
   - Reduce notification frequency
   - Limit Teams startup: Settings → General → Uncheck "Auto-start Teams"

3. **New Teams app** (recommended):
   - Upgrade to "New Teams" (Microsoft Teams 2.0) which uses 50% less memory
   - Available through Settings → "Try the new Teams" toggle

## Prevention
- Regularly clear cache (monthly)
- Keep Teams updated to latest version
- Report persistent issues to vendor management for tracking
