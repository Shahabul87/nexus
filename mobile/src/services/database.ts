/**
 * Local Database Service
 *
 * Provides offline-first storage using expo-sqlite for assessments,
 * patients, and sync queue management.
 *
 * Features:
 * - SQLite-based local storage
 * - Reactive data with Zustand integration
 * - Sync queue for offline operations
 * - Automatic schema migrations
 */

import * as SQLite from 'expo-sqlite';
import { v4 as uuidv4 } from 'uuid';

// ============================================================================
// Types
// ============================================================================

export type PatientType = 'pregnant' | 'newborn';
export type SyncStatus = 'pending' | 'syncing' | 'synced' | 'failed';
export type SeverityLevel = 'RED' | 'YELLOW' | 'GREEN';

export interface Patient {
  id: string;
  type: PatientType;
  externalId?: string;
  name?: string;
  dateOfBirth?: string;
  gestationalWeeks?: number;
  gravida?: number;
  para?: number;
  ageHours?: number;
  birthWeight?: number;
  deliveryType?: string;
  apgarScore?: number;
  createdAt: string;
  updatedAt: string;
}

export interface Assessment {
  id: string;
  patientId: string;
  type: PatientType;
  status: 'draft' | 'complete';
  severity: SeverityLevel;

  // Danger signs
  dangerSignsJson: string;

  // Analysis results
  anemiaResultJson?: string;
  jaundiceResultJson?: string;
  cryResultJson?: string;

  // Workflow results
  workflowResultJson?: string;

  // Files (local URIs)
  conjunctivaImageUri?: string;
  skinImageUri?: string;
  cryAudioUri?: string;

  // Metadata
  syncStatus: SyncStatus;
  syncError?: string;
  createdAt: string;
  updatedAt: string;
  syncedAt?: string;
}

export interface SyncQueueItem {
  id: string;
  entityType: 'patient' | 'assessment';
  entityId: string;
  action: 'create' | 'update' | 'delete';
  payload: string;
  attempts: number;
  lastError?: string;
  createdAt: string;
  processedAt?: string;
}

// ============================================================================
// Database Schema
// ============================================================================

const SCHEMA_VERSION = 1;

const CREATE_PATIENTS_TABLE = `
  CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    external_id TEXT,
    name TEXT,
    date_of_birth TEXT,
    gestational_weeks INTEGER,
    gravida INTEGER,
    para INTEGER,
    age_hours INTEGER,
    birth_weight INTEGER,
    delivery_type TEXT,
    apgar_score INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );
`;

const CREATE_ASSESSMENTS_TABLE = `
  CREATE TABLE IF NOT EXISTS assessments (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    severity TEXT NOT NULL DEFAULT 'GREEN',
    danger_signs_json TEXT,
    anemia_result_json TEXT,
    jaundice_result_json TEXT,
    cry_result_json TEXT,
    workflow_result_json TEXT,
    conjunctiva_image_uri TEXT,
    skin_image_uri TEXT,
    cry_audio_uri TEXT,
    sync_status TEXT NOT NULL DEFAULT 'pending',
    sync_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    synced_at TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
  );
`;

const CREATE_SYNC_QUEUE_TABLE = `
  CREATE TABLE IF NOT EXISTS sync_queue (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    action TEXT NOT NULL,
    payload TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    processed_at TEXT
  );
`;

const CREATE_SCHEMA_VERSION_TABLE = `
  CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
  );
`;

// ============================================================================
// Database Service
// ============================================================================

class DatabaseService {
  private db: SQLite.SQLiteDatabase | null = null;
  private initialized = false;

  /**
   * Initialize the database.
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      this.db = await SQLite.openDatabaseAsync('nexus.db');

      // Create tables
      await this.db.execAsync(CREATE_SCHEMA_VERSION_TABLE);
      await this.db.execAsync(CREATE_PATIENTS_TABLE);
      await this.db.execAsync(CREATE_ASSESSMENTS_TABLE);
      await this.db.execAsync(CREATE_SYNC_QUEUE_TABLE);

      // Check and run migrations
      await this.runMigrations();

      this.initialized = true;
      console.log('Database initialized successfully');
    } catch (error) {
      console.error('Database initialization failed:', error);
      throw error;
    }
  }

  /**
   * Run database migrations.
   */
  private async runMigrations(): Promise<void> {
    if (!this.db) return;

    const result = await this.db.getFirstAsync<{ version: number }>(
      'SELECT version FROM schema_version ORDER BY version DESC LIMIT 1'
    );

    const currentVersion = result?.version || 0;

    if (currentVersion < SCHEMA_VERSION) {
      // Run migrations here as needed
      await this.db.runAsync(
        'INSERT OR REPLACE INTO schema_version (version) VALUES (?)',
        [SCHEMA_VERSION]
      );
    }
  }

  /**
   * Get database instance.
   */
  private getDb(): SQLite.SQLiteDatabase {
    if (!this.db) {
      throw new Error('Database not initialized. Call initialize() first.');
    }
    return this.db;
  }

  // ============================================================================
  // Patient Operations
  // ============================================================================

  /**
   * Create a new patient.
   */
  async createPatient(data: Omit<Patient, 'id' | 'createdAt' | 'updatedAt'>): Promise<Patient> {
    const db = this.getDb();
    const now = new Date().toISOString();
    const patient: Patient = {
      ...data,
      id: uuidv4(),
      createdAt: now,
      updatedAt: now,
    };

    await db.runAsync(
      `INSERT INTO patients (
        id, type, external_id, name, date_of_birth,
        gestational_weeks, gravida, para, age_hours,
        birth_weight, delivery_type, apgar_score,
        created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        patient.id,
        patient.type,
        patient.externalId || null,
        patient.name || null,
        patient.dateOfBirth || null,
        patient.gestationalWeeks || null,
        patient.gravida || null,
        patient.para || null,
        patient.ageHours || null,
        patient.birthWeight || null,
        patient.deliveryType || null,
        patient.apgarScore || null,
        patient.createdAt,
        patient.updatedAt,
      ]
    );

    // Add to sync queue
    await this.addToSyncQueue('patient', patient.id, 'create', patient);

    return patient;
  }

  /**
   * Get patient by ID.
   */
  async getPatient(id: string): Promise<Patient | null> {
    const db = this.getDb();
    const row = await db.getFirstAsync<Record<string, unknown>>(
      'SELECT * FROM patients WHERE id = ?',
      [id]
    );

    return row ? this.mapRowToPatient(row) : null;
  }

  /**
   * Get all patients.
   */
  async getAllPatients(type?: PatientType): Promise<Patient[]> {
    const db = this.getDb();
    const query = type
      ? 'SELECT * FROM patients WHERE type = ? ORDER BY created_at DESC'
      : 'SELECT * FROM patients ORDER BY created_at DESC';
    const params = type ? [type] : [];

    const rows = await db.getAllAsync<Record<string, unknown>>(query, params);
    return rows.map((row) => this.mapRowToPatient(row));
  }

  /**
   * Update patient.
   */
  async updatePatient(id: string, data: Partial<Patient>): Promise<Patient | null> {
    const db = this.getDb();
    const now = new Date().toISOString();

    const updates: string[] = [];
    const values: unknown[] = [];

    if (data.name !== undefined) {
      updates.push('name = ?');
      values.push(data.name);
    }
    if (data.gestationalWeeks !== undefined) {
      updates.push('gestational_weeks = ?');
      values.push(data.gestationalWeeks);
    }
    // Add more fields as needed...

    updates.push('updated_at = ?');
    values.push(now);
    values.push(id);

    await db.runAsync(
      `UPDATE patients SET ${updates.join(', ')} WHERE id = ?`,
      values as (string | number | null)[]
    );

    const updated = await this.getPatient(id);
    if (updated) {
      await this.addToSyncQueue('patient', id, 'update', updated);
    }

    return updated;
  }

  /**
   * Delete patient.
   */
  async deletePatient(id: string): Promise<void> {
    const db = this.getDb();
    await db.runAsync('DELETE FROM patients WHERE id = ?', [id]);
    await this.addToSyncQueue('patient', id, 'delete', { id });
  }

  // ============================================================================
  // Assessment Operations
  // ============================================================================

  /**
   * Create a new assessment.
   */
  async createAssessment(
    data: Omit<Assessment, 'id' | 'syncStatus' | 'createdAt' | 'updatedAt'>
  ): Promise<Assessment> {
    const db = this.getDb();
    const now = new Date().toISOString();
    const assessment: Assessment = {
      ...data,
      id: uuidv4(),
      syncStatus: 'pending',
      createdAt: now,
      updatedAt: now,
    };

    await db.runAsync(
      `INSERT INTO assessments (
        id, patient_id, type, status, severity,
        danger_signs_json, anemia_result_json, jaundice_result_json,
        cry_result_json, workflow_result_json,
        conjunctiva_image_uri, skin_image_uri, cry_audio_uri,
        sync_status, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        assessment.id,
        assessment.patientId,
        assessment.type,
        assessment.status,
        assessment.severity,
        assessment.dangerSignsJson,
        assessment.anemiaResultJson || null,
        assessment.jaundiceResultJson || null,
        assessment.cryResultJson || null,
        assessment.workflowResultJson || null,
        assessment.conjunctivaImageUri || null,
        assessment.skinImageUri || null,
        assessment.cryAudioUri || null,
        assessment.syncStatus,
        assessment.createdAt,
        assessment.updatedAt,
      ]
    );

    // Add to sync queue
    await this.addToSyncQueue('assessment', assessment.id, 'create', assessment);

    return assessment;
  }

  /**
   * Get assessment by ID.
   */
  async getAssessment(id: string): Promise<Assessment | null> {
    const db = this.getDb();
    const row = await db.getFirstAsync<Record<string, unknown>>(
      'SELECT * FROM assessments WHERE id = ?',
      [id]
    );

    return row ? this.mapRowToAssessment(row) : null;
  }

  /**
   * Get assessments for a patient.
   */
  async getPatientAssessments(patientId: string): Promise<Assessment[]> {
    const db = this.getDb();
    const rows = await db.getAllAsync<Record<string, unknown>>(
      'SELECT * FROM assessments WHERE patient_id = ? ORDER BY created_at DESC',
      [patientId]
    );

    return rows.map((row) => this.mapRowToAssessment(row));
  }

  /**
   * Get all assessments.
   */
  async getAllAssessments(type?: PatientType): Promise<Assessment[]> {
    const db = this.getDb();
    const query = type
      ? 'SELECT * FROM assessments WHERE type = ? ORDER BY created_at DESC'
      : 'SELECT * FROM assessments ORDER BY created_at DESC';
    const params = type ? [type] : [];

    const rows = await db.getAllAsync<Record<string, unknown>>(query, params);
    return rows.map((row) => this.mapRowToAssessment(row));
  }

  /**
   * Get assessments by sync status.
   */
  async getAssessmentsBySyncStatus(status: SyncStatus): Promise<Assessment[]> {
    const db = this.getDb();
    const rows = await db.getAllAsync<Record<string, unknown>>(
      'SELECT * FROM assessments WHERE sync_status = ? ORDER BY created_at ASC',
      [status]
    );

    return rows.map((row) => this.mapRowToAssessment(row));
  }

  /**
   * Update assessment.
   */
  async updateAssessment(
    id: string,
    data: Partial<Assessment>
  ): Promise<Assessment | null> {
    const db = this.getDb();
    const now = new Date().toISOString();

    const updates: string[] = [];
    const values: unknown[] = [];

    if (data.status !== undefined) {
      updates.push('status = ?');
      values.push(data.status);
    }
    if (data.severity !== undefined) {
      updates.push('severity = ?');
      values.push(data.severity);
    }
    if (data.dangerSignsJson !== undefined) {
      updates.push('danger_signs_json = ?');
      values.push(data.dangerSignsJson);
    }
    if (data.anemiaResultJson !== undefined) {
      updates.push('anemia_result_json = ?');
      values.push(data.anemiaResultJson);
    }
    if (data.jaundiceResultJson !== undefined) {
      updates.push('jaundice_result_json = ?');
      values.push(data.jaundiceResultJson);
    }
    if (data.cryResultJson !== undefined) {
      updates.push('cry_result_json = ?');
      values.push(data.cryResultJson);
    }
    if (data.workflowResultJson !== undefined) {
      updates.push('workflow_result_json = ?');
      values.push(data.workflowResultJson);
    }
    if (data.syncStatus !== undefined) {
      updates.push('sync_status = ?');
      values.push(data.syncStatus);
    }
    if (data.syncError !== undefined) {
      updates.push('sync_error = ?');
      values.push(data.syncError);
    }
    if (data.syncedAt !== undefined) {
      updates.push('synced_at = ?');
      values.push(data.syncedAt);
    }

    updates.push('updated_at = ?');
    values.push(now);
    values.push(id);

    await db.runAsync(
      `UPDATE assessments SET ${updates.join(', ')} WHERE id = ?`,
      values as (string | number | null)[]
    );

    const updated = await this.getAssessment(id);
    if (updated && data.syncStatus !== 'synced') {
      await this.addToSyncQueue('assessment', id, 'update', updated);
    }

    return updated;
  }

  /**
   * Mark assessment as synced.
   */
  async markAssessmentSynced(id: string): Promise<void> {
    await this.updateAssessment(id, {
      syncStatus: 'synced',
      syncedAt: new Date().toISOString(),
    });
  }

  /**
   * Delete assessment.
   */
  async deleteAssessment(id: string): Promise<void> {
    const db = this.getDb();
    await db.runAsync('DELETE FROM assessments WHERE id = ?', [id]);
    await this.addToSyncQueue('assessment', id, 'delete', { id });
  }

  // ============================================================================
  // Sync Queue Operations
  // ============================================================================

  /**
   * Add item to sync queue.
   */
  async addToSyncQueue<T extends object>(
    entityType: 'patient' | 'assessment',
    entityId: string,
    action: 'create' | 'update' | 'delete',
    payload: T
  ): Promise<SyncQueueItem> {
    const db = this.getDb();
    const now = new Date().toISOString();
    const item: SyncQueueItem = {
      id: uuidv4(),
      entityType,
      entityId,
      action,
      payload: JSON.stringify(payload),
      attempts: 0,
      createdAt: now,
    };

    await db.runAsync(
      `INSERT INTO sync_queue (
        id, entity_type, entity_id, action, payload, attempts, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [item.id, item.entityType, item.entityId, item.action, item.payload, item.attempts, item.createdAt]
    );

    return item;
  }

  /**
   * Get pending sync queue items.
   */
  async getPendingSyncItems(): Promise<SyncQueueItem[]> {
    const db = this.getDb();
    const rows = await db.getAllAsync<Record<string, unknown>>(
      'SELECT * FROM sync_queue WHERE processed_at IS NULL ORDER BY created_at ASC'
    );

    return rows.map((row) => this.mapRowToSyncQueueItem(row));
  }

  /**
   * Mark sync queue item as processed.
   */
  async markSyncItemProcessed(id: string): Promise<void> {
    const db = this.getDb();
    await db.runAsync(
      'UPDATE sync_queue SET processed_at = ? WHERE id = ?',
      [new Date().toISOString(), id]
    );
  }

  /**
   * Update sync queue item with error.
   */
  async updateSyncItemError(id: string, error: string): Promise<void> {
    const db = this.getDb();
    await db.runAsync(
      'UPDATE sync_queue SET attempts = attempts + 1, last_error = ? WHERE id = ?',
      [error, id]
    );
  }

  /**
   * Clear processed sync queue items.
   */
  async clearProcessedSyncItems(): Promise<void> {
    const db = this.getDb();
    await db.runAsync('DELETE FROM sync_queue WHERE processed_at IS NOT NULL');
  }

  /**
   * Get sync queue stats.
   */
  async getSyncStats(): Promise<{
    pending: number;
    processed: number;
    failed: number;
  }> {
    const db = this.getDb();

    const pending = await db.getFirstAsync<{ count: number }>(
      'SELECT COUNT(*) as count FROM sync_queue WHERE processed_at IS NULL'
    );

    const processed = await db.getFirstAsync<{ count: number }>(
      'SELECT COUNT(*) as count FROM sync_queue WHERE processed_at IS NOT NULL'
    );

    const failed = await db.getFirstAsync<{ count: number }>(
      'SELECT COUNT(*) as count FROM sync_queue WHERE attempts >= 3 AND processed_at IS NULL'
    );

    return {
      pending: pending?.count || 0,
      processed: processed?.count || 0,
      failed: failed?.count || 0,
    };
  }

  // ============================================================================
  // Helper Methods
  // ============================================================================

  private mapRowToPatient(row: Record<string, unknown>): Patient {
    return {
      id: row.id as string,
      type: row.type as PatientType,
      externalId: row.external_id as string | undefined,
      name: row.name as string | undefined,
      dateOfBirth: row.date_of_birth as string | undefined,
      gestationalWeeks: row.gestational_weeks as number | undefined,
      gravida: row.gravida as number | undefined,
      para: row.para as number | undefined,
      ageHours: row.age_hours as number | undefined,
      birthWeight: row.birth_weight as number | undefined,
      deliveryType: row.delivery_type as string | undefined,
      apgarScore: row.apgar_score as number | undefined,
      createdAt: row.created_at as string,
      updatedAt: row.updated_at as string,
    };
  }

  private mapRowToAssessment(row: Record<string, unknown>): Assessment {
    return {
      id: row.id as string,
      patientId: row.patient_id as string,
      type: row.type as PatientType,
      status: row.status as 'draft' | 'complete',
      severity: row.severity as SeverityLevel,
      dangerSignsJson: row.danger_signs_json as string,
      anemiaResultJson: row.anemia_result_json as string | undefined,
      jaundiceResultJson: row.jaundice_result_json as string | undefined,
      cryResultJson: row.cry_result_json as string | undefined,
      workflowResultJson: row.workflow_result_json as string | undefined,
      conjunctivaImageUri: row.conjunctiva_image_uri as string | undefined,
      skinImageUri: row.skin_image_uri as string | undefined,
      cryAudioUri: row.cry_audio_uri as string | undefined,
      syncStatus: row.sync_status as SyncStatus,
      syncError: row.sync_error as string | undefined,
      createdAt: row.created_at as string,
      updatedAt: row.updated_at as string,
      syncedAt: row.synced_at as string | undefined,
    };
  }

  private mapRowToSyncQueueItem(row: Record<string, unknown>): SyncQueueItem {
    return {
      id: row.id as string,
      entityType: row.entity_type as 'patient' | 'assessment',
      entityId: row.entity_id as string,
      action: row.action as 'create' | 'update' | 'delete',
      payload: row.payload as string,
      attempts: row.attempts as number,
      lastError: row.last_error as string | undefined,
      createdAt: row.created_at as string,
      processedAt: row.processed_at as string | undefined,
    };
  }

  /**
   * Close the database connection.
   */
  async close(): Promise<void> {
    if (this.db) {
      await this.db.closeAsync();
      this.db = null;
      this.initialized = false;
    }
  }

  /**
   * Reset the database (for testing/development).
   */
  async reset(): Promise<void> {
    const db = this.getDb();
    await db.execAsync('DELETE FROM sync_queue');
    await db.execAsync('DELETE FROM assessments');
    await db.execAsync('DELETE FROM patients');
    console.log('Database reset complete');
  }
}

// Export singleton instance
export const database = new DatabaseService();
