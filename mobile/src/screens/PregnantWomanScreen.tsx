/**
 * Pregnant Woman Assessment Screen
 *
 * Comprehensive maternal health assessment flow
 * Uses HAI-DEF models for screening
 *
 * Flow:
 * 1. Patient info collection
 * 2. Symptom checklist (WHO IMNCI danger signs)
 * 3. Anemia screening (conjunctiva photo)
 * 4. Agentic synthesis with MedGemma
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
  Alert,
  ActivityIndicator,
  TextInput,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import type { RootStackParamList } from '../../App';
import { edgeAI } from '../services/edgeAI';
import { analyzeAnemia, runCombinedAssessment } from '../services/nexusApi';

type PregnantWomanScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'PregnantWoman'>;
};

interface PatientInfo {
  gestationalWeeks: string;
  gravida: string;
  para: string;
}

interface DangerSign {
  id: string;
  label: string;
  description: string;
  severity: 'high' | 'medium';
}

const DANGER_SIGNS: DangerSign[] = [
  {
    id: 'vaginal_bleeding',
    label: 'Vaginal Bleeding',
    description: 'Any vaginal bleeding during pregnancy',
    severity: 'high',
  },
  {
    id: 'severe_headache',
    label: 'Severe Headache',
    description: 'Headache with blurred vision',
    severity: 'high',
  },
  {
    id: 'convulsions',
    label: 'Convulsions',
    description: 'Seizures or fits',
    severity: 'high',
  },
  {
    id: 'swelling',
    label: 'Swelling of Face/Hands',
    description: 'Sudden swelling (pre-eclampsia sign)',
    severity: 'high',
  },
  {
    id: 'fever',
    label: 'Fever',
    description: 'Temperature above 38C',
    severity: 'medium',
  },
  {
    id: 'abdominal_pain',
    label: 'Severe Abdominal Pain',
    description: 'Persistent or severe pain',
    severity: 'medium',
  },
  {
    id: 'reduced_movement',
    label: 'Reduced Fetal Movement',
    description: 'Baby moving less than usual',
    severity: 'medium',
  },
  {
    id: 'weakness',
    label: 'Severe Weakness/Fatigue',
    description: 'Unable to perform daily activities',
    severity: 'medium',
  },
];

type Step = 'info' | 'symptoms' | 'anemia' | 'analysis';

export default function PregnantWomanScreen({
  navigation,
}: PregnantWomanScreenProps): React.JSX.Element {
  const [currentStep, setCurrentStep] = useState<Step>('info');
  const [patientInfo, setPatientInfo] = useState<PatientInfo>({
    gestationalWeeks: '',
    gravida: '',
    para: '',
  });
  const [selectedSigns, setSelectedSigns] = useState<string[]>([]);
  const [conjunctivaImage, setConjunctivaImage] = useState<string | null>(null);
  const [_isAnalyzing, setIsAnalyzing] = useState(false);
  const [useEdgeAI, _setUseEdgeAI] = useState(false);

  const toggleDangerSign = useCallback((signId: string) => {
    setSelectedSigns((prev) =>
      prev.includes(signId)
        ? prev.filter((id) => id !== signId)
        : [...prev, signId]
    );
  }, []);

  const captureConjunctiva = async (): Promise<void> => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();

    if (!permissionResult.granted) {
      Alert.alert('Permission Required', 'Please allow camera access.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setConjunctivaImage(result.assets[0].uri);
    }
  };

  const runAssessment = async (): Promise<void> => {
    setIsAnalyzing(true);
    setCurrentStep('analysis');

    try {
      // Run anemia analysis
      let anemiaResult;
      if (useEdgeAI && edgeAI.isEdgeAvailable()) {
        anemiaResult = conjunctivaImage
          ? await edgeAI.analyzeAnemia(conjunctivaImage)
          : null;
      } else {
        anemiaResult = conjunctivaImage
          ? await analyzeAnemia(conjunctivaImage)
          : null;
      }

      // Build symptom summary
      const symptomSummary = selectedSigns
        .map((id) => DANGER_SIGNS.find((s) => s.id === id)?.label)
        .filter(Boolean)
        .join(', ');

      // Run combined assessment with MedGemma
      const combinedResult = await runCombinedAssessment({
        conjunctivaImage,
        skinImage: null,
        cryAudio: null,
      });

      // Compile results
      const results = {
        patient_type: 'pregnant_woman',
        gestational_weeks: patientInfo.gestationalWeeks,
        gravida: patientInfo.gravida,
        para: patientInfo.para,
        danger_signs: selectedSigns,
        danger_signs_count: selectedSigns.length,
        has_high_severity_signs: selectedSigns.some(
          (id) => DANGER_SIGNS.find((s) => s.id === id)?.severity === 'high'
        ),
        anemia: anemiaResult,
        symptoms_reported: symptomSummary,
        ...combinedResult,
        severity_level: determineSeverity(selectedSigns, anemiaResult as Record<string, unknown> | null),
      };

      navigation.navigate('Results', { type: 'combined', results });
    } catch (error) {
      Alert.alert('Error', 'Failed to complete assessment. Please try again.');
      console.error('Assessment error:', error);
      setCurrentStep('anemia');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const determineSeverity = (
    signs: string[],
    anemia: Record<string, unknown> | null
  ): string => {
    const hasHighSeveritySigns = signs.some(
      (id) => DANGER_SIGNS.find((s) => s.id === id)?.severity === 'high'
    );

    if (hasHighSeveritySigns) return 'RED';
    if (signs.length > 2) return 'YELLOW';
    if (anemia && (anemia as { is_anemic?: boolean }).is_anemic) return 'YELLOW';
    return 'GREEN';
  };

  const canProceed = (): boolean => {
    switch (currentStep) {
      case 'info':
        return patientInfo.gestationalWeeks.length > 0;
      case 'symptoms':
        return true;
      case 'anemia':
        return true;
      default:
        return false;
    }
  };

  const nextStep = (): void => {
    const steps: Step[] = ['info', 'symptoms', 'anemia', 'analysis'];
    const currentIndex = steps.indexOf(currentStep);
    if (currentIndex < steps.length - 1) {
      setCurrentStep(steps[currentIndex + 1]);
    }
  };

  const prevStep = (): void => {
    const steps: Step[] = ['info', 'symptoms', 'anemia', 'analysis'];
    const currentIndex = steps.indexOf(currentStep);
    if (currentIndex > 0) {
      setCurrentStep(steps[currentIndex - 1]);
    }
  };

  const getStepNumber = (): number => {
    const steps: Step[] = ['info', 'symptoms', 'anemia', 'analysis'];
    return steps.indexOf(currentStep) + 1;
  };

  const renderPatientInfo = (): React.JSX.Element => (
    <View style={styles.stepContent}>
      <Text style={styles.stepTitle}>Patient Information</Text>
      <Text style={styles.stepDescription}>
        Enter basic pregnancy information
      </Text>

      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>Gestational Age (weeks)</Text>
        <TextInput
          style={styles.input}
          placeholder="e.g., 28"
          keyboardType="numeric"
          value={patientInfo.gestationalWeeks}
          onChangeText={(text) =>
            setPatientInfo((prev) => ({ ...prev, gestationalWeeks: text }))
          }
        />
      </View>

      <View style={styles.inputRow}>
        <View style={[styles.inputGroup, { flex: 1, marginRight: 8 }]}>
          <Text style={styles.inputLabel}>Gravida (pregnancies)</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g., 2"
            keyboardType="numeric"
            value={patientInfo.gravida}
            onChangeText={(text) =>
              setPatientInfo((prev) => ({ ...prev, gravida: text }))
            }
          />
        </View>
        <View style={[styles.inputGroup, { flex: 1, marginLeft: 8 }]}>
          <Text style={styles.inputLabel}>Para (births)</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g., 1"
            keyboardType="numeric"
            value={patientInfo.para}
            onChangeText={(text) =>
              setPatientInfo((prev) => ({ ...prev, para: text }))
            }
          />
        </View>
      </View>
    </View>
  );

  const renderSymptoms = (): React.JSX.Element => (
    <View style={styles.stepContent}>
      <Text style={styles.stepTitle}>Danger Signs Checklist</Text>
      <Text style={styles.stepDescription}>
        Check any symptoms the patient is experiencing (WHO IMNCI)
      </Text>

      <View style={styles.symptomsContainer}>
        {DANGER_SIGNS.map((sign) => (
          <TouchableOpacity
            key={sign.id}
            style={[
              styles.symptomCard,
              selectedSigns.includes(sign.id) && styles.symptomCardSelected,
              sign.severity === 'high' && styles.symptomCardHighSeverity,
            ]}
            onPress={() => toggleDangerSign(sign.id)}
          >
            <View style={styles.symptomCheckbox}>
              {selectedSigns.includes(sign.id) ? (
                <Ionicons name="checkbox" size={24} color="#1a73e8" />
              ) : (
                <Ionicons name="square-outline" size={24} color="#999" />
              )}
            </View>
            <View style={styles.symptomInfo}>
              <Text style={styles.symptomLabel}>{sign.label}</Text>
              <Text style={styles.symptomDescription}>{sign.description}</Text>
            </View>
            {sign.severity === 'high' && (
              <View style={styles.severityBadge}>
                <Text style={styles.severityBadgeText}>URGENT</Text>
              </View>
            )}
          </TouchableOpacity>
        ))}
      </View>

      {selectedSigns.length > 0 && (
        <View style={styles.selectedCount}>
          <Ionicons name="warning" size={20} color="#856404" />
          <Text style={styles.selectedCountText}>
            {selectedSigns.length} danger sign(s) selected
          </Text>
        </View>
      )}
    </View>
  );

  const renderAnemiaCapture = (): React.JSX.Element => (
    <View style={styles.stepContent}>
      <Text style={styles.stepTitle}>Anemia Screening</Text>
      <Text style={styles.stepDescription}>
        Capture a photo of the inner eyelid (conjunctiva) for anemia detection
      </Text>

      <View style={styles.captureContainer}>
        {conjunctivaImage ? (
          <Image source={{ uri: conjunctivaImage }} style={styles.capturedImage} />
        ) : (
          <View style={styles.capturePlaceholder}>
            <Ionicons name="eye-outline" size={64} color="#ccc" />
            <Text style={styles.capturePlaceholderText}>No image captured</Text>
          </View>
        )}
      </View>

      <TouchableOpacity style={styles.captureButton} onPress={captureConjunctiva}>
        <Ionicons name="camera" size={24} color="#fff" />
        <Text style={styles.captureButtonText}>
          {conjunctivaImage ? 'Retake Photo' : 'Take Photo'}
        </Text>
      </TouchableOpacity>

      <View style={styles.captureInstructions}>
        <Text style={styles.instructionsTitle}>Instructions:</Text>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={16} color="#28a745" />
          <Text style={styles.instructionText}>
            Gently pull down the lower eyelid
          </Text>
        </View>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={16} color="#28a745" />
          <Text style={styles.instructionText}>
            Ensure good lighting (natural light preferred)
          </Text>
        </View>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={16} color="#28a745" />
          <Text style={styles.instructionText}>
            Focus on the pink inner eyelid area
          </Text>
        </View>
      </View>

      <View style={styles.skipNote}>
        <Text style={styles.skipNoteText}>
          Photo is optional but recommended for complete assessment
        </Text>
      </View>
    </View>
  );

  const renderAnalysis = (): React.JSX.Element => (
    <View style={styles.stepContent}>
      <View style={styles.analysisContainer}>
        <ActivityIndicator size="large" color="#1a73e8" />
        <Text style={styles.analysisTitle}>Running Assessment</Text>
        <Text style={styles.analysisDescription}>
          Analyzing with HAI-DEF models...
        </Text>

        <View style={styles.analysisSteps}>
          <View style={styles.analysisStep}>
            <Ionicons name="checkmark-circle" size={20} color="#28a745" />
            <Text style={styles.analysisStepText}>Patient info collected</Text>
          </View>
          <View style={styles.analysisStep}>
            <Ionicons name="checkmark-circle" size={20} color="#28a745" />
            <Text style={styles.analysisStepText}>
              {selectedSigns.length} danger signs evaluated
            </Text>
          </View>
          <View style={styles.analysisStep}>
            {conjunctivaImage ? (
              <Ionicons name="hourglass" size={20} color="#1a73e8" />
            ) : (
              <Ionicons name="remove-circle" size={20} color="#999" />
            )}
            <Text style={styles.analysisStepText}>
              {conjunctivaImage ? 'Anemia screening with MedSigLIP...' : 'Anemia screening skipped'}
            </Text>
          </View>
          <View style={styles.analysisStep}>
            <Ionicons name="hourglass" size={20} color="#1a73e8" />
            <Text style={styles.analysisStepText}>
              Clinical synthesis with MedGemma...
            </Text>
          </View>
        </View>
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      {/* Progress Bar */}
      <View style={styles.progressBar}>
        <View style={styles.progressTrack}>
          <View
            style={[styles.progressFill, { width: `${(getStepNumber() / 4) * 100}%` }]}
          />
        </View>
        <Text style={styles.progressText}>Step {getStepNumber()} of 4</Text>
      </View>

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        {currentStep === 'info' && renderPatientInfo()}
        {currentStep === 'symptoms' && renderSymptoms()}
        {currentStep === 'anemia' && renderAnemiaCapture()}
        {currentStep === 'analysis' && renderAnalysis()}
      </ScrollView>

      {/* Navigation Buttons */}
      {currentStep !== 'analysis' && (
        <View style={styles.navigationButtons}>
          {currentStep !== 'info' && (
            <TouchableOpacity style={styles.backButton} onPress={prevStep}>
              <Ionicons name="arrow-back" size={24} color="#1a73e8" />
              <Text style={styles.backButtonText}>Back</Text>
            </TouchableOpacity>
          )}

          <TouchableOpacity
            style={[
              styles.nextButton,
              currentStep === 'anemia' && styles.submitButton,
              !canProceed() && styles.nextButtonDisabled,
            ]}
            onPress={currentStep === 'anemia' ? runAssessment : nextStep}
            disabled={!canProceed()}
          >
            <Text style={styles.nextButtonText}>
              {currentStep === 'anemia' ? 'Run Assessment' : 'Next'}
            </Text>
            <Ionicons
              name={currentStep === 'anemia' ? 'analytics' : 'arrow-forward'}
              size={24}
              color="#fff"
            />
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  progressBar: {
    backgroundColor: '#fff',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  progressTrack: {
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#1a73e8',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 100,
  },
  stepContent: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  stepDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 24,
  },
  inputGroup: {
    marginBottom: 16,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
  },
  inputRow: {
    flexDirection: 'row',
  },
  symptomsContainer: {
    gap: 12,
  },
  symptomCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  symptomCardSelected: {
    borderColor: '#1a73e8',
    backgroundColor: '#e8f4fd',
  },
  symptomCardHighSeverity: {
    borderLeftWidth: 4,
    borderLeftColor: '#e74c3c',
  },
  symptomCheckbox: {
    marginRight: 12,
  },
  symptomInfo: {
    flex: 1,
  },
  symptomLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  symptomDescription: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  severityBadge: {
    backgroundColor: '#f8d7da',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  severityBadgeText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#721c24',
  },
  selectedCount: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff3cd',
    padding: 12,
    borderRadius: 8,
    marginTop: 16,
  },
  selectedCountText: {
    marginLeft: 8,
    fontSize: 14,
    color: '#856404',
    fontWeight: '600',
  },
  captureContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    overflow: 'hidden',
    aspectRatio: 4 / 3,
    marginBottom: 16,
  },
  capturedImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  capturePlaceholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f9f9f9',
  },
  capturePlaceholderText: {
    marginTop: 8,
    fontSize: 14,
    color: '#999',
  },
  captureButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#e74c3c',
    paddingVertical: 16,
    borderRadius: 12,
    marginBottom: 24,
  },
  captureButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 8,
  },
  captureInstructions: {
    backgroundColor: '#e8f5e9',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
  },
  instructionsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2e7d32',
    marginBottom: 8,
  },
  instructionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  instructionText: {
    marginLeft: 8,
    fontSize: 13,
    color: '#333',
  },
  skipNote: {
    alignItems: 'center',
  },
  skipNoteText: {
    fontSize: 12,
    color: '#999',
    fontStyle: 'italic',
  },
  analysisContainer: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  analysisTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginTop: 24,
    marginBottom: 8,
  },
  analysisDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 32,
  },
  analysisSteps: {
    alignSelf: 'stretch',
  },
  analysisStep: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    backgroundColor: '#fff',
    borderRadius: 8,
    marginBottom: 8,
  },
  analysisStepText: {
    marginLeft: 12,
    fontSize: 14,
    color: '#333',
  },
  navigationButtons: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    padding: 16,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    paddingHorizontal: 24,
    marginRight: 12,
  },
  backButtonText: {
    marginLeft: 8,
    fontSize: 16,
    color: '#1a73e8',
    fontWeight: '600',
  },
  nextButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1a73e8',
    paddingVertical: 16,
    borderRadius: 12,
  },
  submitButton: {
    backgroundColor: '#28a745',
  },
  nextButtonDisabled: {
    backgroundColor: '#ccc',
  },
  nextButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginRight: 8,
  },
});
