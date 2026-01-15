/**
 * NEXUS Services Index
 *
 * Central export for all service modules.
 */

// Edge AI Service
export { edgeAI } from './edgeAI';
export type { AnemiaResult, JaundiceResult, CryResult } from './edgeAI';

// Cloud API Service
export { nexusApi, synthesize, analyzeAnemia, analyzeJaundice, analyzeCry, runCombinedAssessment, checkApiHealth } from './nexusApi';
export type { SynthesisResult } from './nexusApi';

// Agentic Workflow Engine
export { AgenticWorkflowEngine, agenticWorkflow, createWorkflowEngine } from './agenticWorkflow';
export type {
  WorkflowInput,
  WorkflowResult,
  WorkflowState,
  PatientType,
  DangerSign,
  PatientInfo,
  TriageResult,
  ImageAnalysisResult,
  AudioAnalysisResult,
  ProtocolResult,
  ReferralResult,
  SeverityLevel,
} from './agenticWorkflow';

// Database Service
export { database } from './database';
export type {
  Patient,
  Assessment,
  SyncQueueItem,
  SyncStatus,
} from './database';

// Sync Service
export { syncService } from './syncService';
export type { SyncState, SyncProgress, SyncResult } from './syncService';
