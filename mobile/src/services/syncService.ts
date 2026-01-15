/**
 * Sync Service
 *
 * Handles background synchronization of offline data with the server.
 * Uses the sync queue from the database service.
 *
 * Features:
 * - Network state monitoring
 * - Automatic sync when online
 * - Retry logic with exponential backoff
 * - Conflict resolution
 * - File upload for images/audio
 */

import * as Network from 'expo-network';
import * as FileSystem from 'expo-file-system';
import { database, SyncQueueItem, Assessment, Patient } from './database';

// ============================================================================
// Types
// ============================================================================

export type SyncState = 'idle' | 'syncing' | 'error' | 'offline';

export interface SyncProgress {
  total: number;
  completed: number;
  failed: number;
  current?: string;
}

export interface SyncResult {
  success: boolean;
  synced: number;
  failed: number;
  errors: string[];
}

type SyncCallback = (state: SyncState, progress?: SyncProgress) => void;

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000';
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1000;
const SYNC_INTERVAL_MS = 60000; // 1 minute

// ============================================================================
// Sync Service
// ============================================================================

class SyncService {
  private state: SyncState = 'idle';
  private isOnline = true;
  private syncInProgress = false;
  private syncTimer: NodeJS.Timeout | null = null;
  private networkCheckTimer: NodeJS.Timeout | null = null;
  private listeners: Set<SyncCallback> = new Set();

  /**
   * Initialize the sync service.
   */
  async initialize(): Promise<void> {
    // Check initial network state
    await this.checkNetworkState();
    this.updateState(this.isOnline ? 'idle' : 'offline');

    // Start periodic network check (since expo-network doesn't have listeners)
    this.startNetworkCheck();

    // Start periodic sync
    this.startPeriodicSync();

    console.log('Sync service initialized');
  }

  /**
   * Check current network state.
   */
  private async checkNetworkState(): Promise<void> {
    try {
      const networkState = await Network.getNetworkStateAsync();
      const wasOnline = this.isOnline;
      this.isOnline = networkState.isConnected ?? false;

      if (!wasOnline && this.isOnline) {
        // Just came online, trigger sync
        console.log('Device came online, triggering sync');
        this.sync();
      } else if (wasOnline && !this.isOnline) {
        console.log('Device went offline');
        this.updateState('offline');
      }
    } catch (error) {
      console.error('Network check error:', error);
      // Assume online if check fails
      this.isOnline = true;
    }
  }

  /**
   * Start periodic network state check.
   */
  private startNetworkCheck(): void {
    if (this.networkCheckTimer) {
      clearInterval(this.networkCheckTimer);
    }

    // Check network state every 5 seconds
    this.networkCheckTimer = setInterval(() => {
      this.checkNetworkState();
    }, 5000);
  }

  /**
   * Stop network state check.
   */
  private stopNetworkCheck(): void {
    if (this.networkCheckTimer) {
      clearInterval(this.networkCheckTimer);
      this.networkCheckTimer = null;
    }
  }

  /**
   * Start periodic sync timer.
   */
  private startPeriodicSync(): void {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
    }

    this.syncTimer = setInterval(() => {
      if (this.isOnline && !this.syncInProgress) {
        this.sync();
      }
    }, SYNC_INTERVAL_MS);
  }

  /**
   * Stop periodic sync timer.
   */
  private stopPeriodicSync(): void {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
      this.syncTimer = null;
    }
  }

  /**
   * Update sync state and notify listeners.
   */
  private updateState(newState: SyncState, progress?: SyncProgress): void {
    this.state = newState;
    this.listeners.forEach((callback) => callback(newState, progress));
  }

  /**
   * Subscribe to sync state changes.
   */
  subscribe(callback: SyncCallback): () => void {
    this.listeners.add(callback);
    // Immediately notify of current state
    callback(this.state);
    return () => this.listeners.delete(callback);
  }

  /**
   * Get current sync state.
   */
  getState(): SyncState {
    return this.state;
  }

  /**
   * Check if online.
   */
  checkOnline(): boolean {
    return this.isOnline;
  }

  /**
   * Manually trigger sync.
   */
  async sync(): Promise<SyncResult> {
    if (!this.isOnline) {
      return { success: false, synced: 0, failed: 0, errors: ['Device is offline'] };
    }

    if (this.syncInProgress) {
      return { success: false, synced: 0, failed: 0, errors: ['Sync already in progress'] };
    }

    this.syncInProgress = true;
    this.updateState('syncing');

    const result: SyncResult = {
      success: true,
      synced: 0,
      failed: 0,
      errors: [],
    };

    try {
      // Get pending sync items
      const items = await database.getPendingSyncItems();

      if (items.length === 0) {
        this.updateState('idle');
        this.syncInProgress = false;
        return result;
      }

      const progress: SyncProgress = {
        total: items.length,
        completed: 0,
        failed: 0,
      };

      this.updateState('syncing', progress);

      // Process each item
      for (const item of items) {
        try {
          progress.current = `${item.entityType}:${item.entityId}`;
          this.updateState('syncing', progress);

          await this.processSyncItem(item);

          progress.completed++;
          result.synced++;

          // Mark as processed
          await database.markSyncItemProcessed(item.id);

          // Update entity sync status
          if (item.entityType === 'assessment') {
            await database.markAssessmentSynced(item.entityId);
          }
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Unknown error';
          progress.failed++;
          result.failed++;
          result.errors.push(`${item.entityType}:${item.entityId}: ${errorMsg}`);

          // Update item with error
          await database.updateSyncItemError(item.id, errorMsg);

          // Check if max retries exceeded
          if (item.attempts >= MAX_RETRIES - 1) {
            // Mark assessment as failed
            if (item.entityType === 'assessment') {
              await database.updateAssessment(item.entityId, {
                syncStatus: 'failed',
                syncError: errorMsg,
              });
            }
          }
        }

        this.updateState('syncing', progress);
      }

      result.success = result.failed === 0;
      this.updateState(result.failed > 0 ? 'error' : 'idle');
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      result.success = false;
      result.errors.push(errorMsg);
      this.updateState('error');
    } finally {
      this.syncInProgress = false;
    }

    return result;
  }

  /**
   * Process a single sync queue item.
   */
  private async processSyncItem(item: SyncQueueItem): Promise<void> {
    const payload = JSON.parse(item.payload);

    switch (item.entityType) {
      case 'patient':
        await this.syncPatient(item.action, payload);
        break;
      case 'assessment':
        await this.syncAssessment(item.action, payload);
        break;
      default:
        throw new Error(`Unknown entity type: ${item.entityType}`);
    }
  }

  /**
   * Sync patient to server.
   */
  private async syncPatient(
    action: 'create' | 'update' | 'delete',
    patient: Patient
  ): Promise<void> {
    const endpoint = `${API_BASE_URL}/api/patients`;

    switch (action) {
      case 'create':
        await this.fetchWithRetry(`${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(patient),
        });
        break;
      case 'update':
        await this.fetchWithRetry(`${endpoint}/${patient.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(patient),
        });
        break;
      case 'delete':
        await this.fetchWithRetry(`${endpoint}/${patient.id}`, {
          method: 'DELETE',
        });
        break;
    }
  }

  /**
   * Sync assessment to server.
   */
  private async syncAssessment(
    action: 'create' | 'update' | 'delete',
    assessment: Assessment
  ): Promise<void> {
    const endpoint = `${API_BASE_URL}/api/assessments`;

    // Upload any associated files first
    const fileUrls: Record<string, string> = {};

    if (assessment.conjunctivaImageUri) {
      fileUrls.conjunctivaImage = await this.uploadFile(
        assessment.conjunctivaImageUri,
        'image/jpeg'
      );
    }
    if (assessment.skinImageUri) {
      fileUrls.skinImage = await this.uploadFile(
        assessment.skinImageUri,
        'image/jpeg'
      );
    }
    if (assessment.cryAudioUri) {
      fileUrls.cryAudio = await this.uploadFile(
        assessment.cryAudioUri,
        'audio/wav'
      );
    }

    // Prepare assessment data with file URLs
    const assessmentData = {
      ...assessment,
      conjunctivaImageUrl: fileUrls.conjunctivaImage,
      skinImageUrl: fileUrls.skinImage,
      cryAudioUrl: fileUrls.cryAudio,
    };

    switch (action) {
      case 'create':
        await this.fetchWithRetry(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(assessmentData),
        });
        break;
      case 'update':
        await this.fetchWithRetry(`${endpoint}/${assessment.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(assessmentData),
        });
        break;
      case 'delete':
        await this.fetchWithRetry(`${endpoint}/${assessment.id}`, {
          method: 'DELETE',
        });
        break;
    }
  }

  /**
   * Upload a file to the server.
   */
  private async uploadFile(uri: string, mimeType: string): Promise<string> {
    // Check if file exists
    const fileInfo = await FileSystem.getInfoAsync(uri);
    if (!fileInfo.exists) {
      throw new Error(`File not found: ${uri}`);
    }

    // Read file as base64
    const base64 = await FileSystem.readAsStringAsync(uri, {
      encoding: FileSystem.EncodingType.Base64,
    });

    // Upload to server
    const response = await this.fetchWithRetry(`${API_BASE_URL}/api/upload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        data: base64,
        mimeType,
        filename: uri.split('/').pop(),
      }),
    });

    const result = await response.json();
    return result.url;
  }

  /**
   * Fetch with retry logic.
   */
  private async fetchWithRetry(
    url: string,
    options: RequestInit,
    retries = MAX_RETRIES
  ): Promise<Response> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < retries; attempt++) {
      try {
        const response = await fetch(url, {
          ...options,
          signal: AbortSignal.timeout(30000), // 30 second timeout
        });

        if (!response.ok) {
          throw new Error(`HTTP error: ${response.status} ${response.statusText}`);
        }

        return response;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error('Unknown error');

        if (attempt < retries - 1) {
          // Wait before retry with exponential backoff
          const delay = RETRY_DELAY_MS * Math.pow(2, attempt);
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError || new Error('Max retries exceeded');
  }

  /**
   * Force sync all pending assessments.
   */
  async forceSyncAll(): Promise<SyncResult> {
    // Reset failed items to pending
    const items = await database.getPendingSyncItems();
    for (const item of items) {
      if (item.attempts >= MAX_RETRIES) {
        // Reset attempts for retry
        await database.addToSyncQueue(
          item.entityType,
          item.entityId,
          item.action,
          JSON.parse(item.payload)
        );
        await database.markSyncItemProcessed(item.id);
      }
    }

    return this.sync();
  }

  /**
   * Get sync queue statistics.
   */
  async getStats(): Promise<{
    pending: number;
    processed: number;
    failed: number;
    isOnline: boolean;
  }> {
    const dbStats = await database.getSyncStats();
    return {
      ...dbStats,
      isOnline: this.isOnline,
    };
  }

  /**
   * Clean up resources.
   */
  cleanup(): void {
    this.stopPeriodicSync();
    this.stopNetworkCheck();
    this.listeners.clear();
  }
}

// Export singleton instance
export const syncService = new SyncService();
