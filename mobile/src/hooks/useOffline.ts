/**
 * useOffline Hook
 *
 * Provides offline status and sync state management for React Native components.
 */

import { useState, useEffect, useCallback } from 'react';
import { syncService, SyncState, SyncProgress, SyncResult } from '../services';
import { database } from '../services/database';

export interface OfflineState {
  isOnline: boolean;
  syncState: SyncState;
  syncProgress?: SyncProgress;
  pendingCount: number;
}

export interface UseOfflineResult extends OfflineState {
  sync: () => Promise<SyncResult>;
  forceSync: () => Promise<SyncResult>;
  refreshPendingCount: () => Promise<void>;
}

/**
 * Hook for managing offline state and sync operations.
 */
export function useOffline(): UseOfflineResult {
  const [state, setState] = useState<OfflineState>({
    isOnline: true,
    syncState: 'idle',
    pendingCount: 0,
  });

  // Subscribe to sync state changes
  useEffect(() => {
    const unsubscribe = syncService.subscribe((syncState, progress) => {
      setState((prev) => ({
        ...prev,
        isOnline: syncService.checkOnline(),
        syncState,
        syncProgress: progress,
      }));
    });

    // Initial state
    setState((prev) => ({
      ...prev,
      isOnline: syncService.checkOnline(),
      syncState: syncService.getState(),
    }));

    return () => {
      unsubscribe();
    };
  }, []);

  // Refresh pending count
  const refreshPendingCount = useCallback(async () => {
    try {
      const stats = await database.getSyncStats();
      setState((prev) => ({
        ...prev,
        pendingCount: stats.pending,
      }));
    } catch (error) {
      console.error('Failed to get pending count:', error);
    }
  }, []);

  // Refresh pending count periodically
  useEffect(() => {
    refreshPendingCount();
    const interval = setInterval(refreshPendingCount, 10000);
    return () => clearInterval(interval);
  }, [refreshPendingCount]);

  // Trigger sync
  const sync = useCallback(async (): Promise<SyncResult> => {
    const result = await syncService.sync();
    await refreshPendingCount();
    return result;
  }, [refreshPendingCount]);

  // Force sync all
  const forceSync = useCallback(async (): Promise<SyncResult> => {
    const result = await syncService.forceSyncAll();
    await refreshPendingCount();
    return result;
  }, [refreshPendingCount]);

  return {
    ...state,
    sync,
    forceSync,
    refreshPendingCount,
  };
}

/**
 * Hook for checking network status only.
 */
export function useNetworkStatus(): boolean {
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    const unsubscribe = syncService.subscribe(() => {
      setIsOnline(syncService.checkOnline());
    });

    setIsOnline(syncService.checkOnline());

    return () => {
      unsubscribe();
    };
  }, []);

  return isOnline;
}
