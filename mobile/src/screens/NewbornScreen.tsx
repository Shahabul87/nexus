/**
 * Newborn Assessment Screen
 *
 * Comprehensive neonatal health assessment flow
 * Uses HAI-DEF models: MedSigLIP for jaundice, HeAR for cry analysis
 *
 * Flow:
 * 1. Newborn info collection
 * 2. Danger signs checklist (WHO IMNCI)
 * 3. Jaundice screening (skin photo)
 * 4. Cry analysis (audio recording)
 * 5. Agentic synthesis with MedGemma
 */

import React, { useState, useCallback, useRef } from 'react';
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
import { Audio } from 'expo-av';
import { Ionicons } from '@expo/vector-icons';
import type { RootStackParamList } from '../../App';
import { edgeAI } from '../services/edgeAI';
import {
  analyzeJaundice,
  analyzeCry,
  runCombinedAssessment,
} from '../services/nexusApi';

type NewbornScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Newborn'>;
};

interface NewbornInfo {
  ageHours: string;
  birthWeight: string;
  deliveryType: 'vaginal' | 'cesarean' | '';
  apgarScore: string;
}

interface DangerSign {
  id: string;
  label: string;
  description: string;
  severity: 'critical' | 'high' | 'medium';
}

const DANGER_SIGNS: DangerSign[] = [
  {
    id: 'not_breathing',
    label: 'Not Breathing/Gasping',
    description: 'Apnea or irregular breathing',
    severity: 'critical',
  },
  {
    id: 'convulsions',
    label: 'Convulsions',
    description: 'Seizures or abnormal movements',
    severity: 'critical',
  },
  {
    id: 'severe_chest_indrawing',
    label: 'Severe Chest Indrawing',
    description: 'Significant respiratory distress',
    severity: 'critical',
  },
  {
    id: 'no_movement',
    label: 'No Movement/Limp',
    description: 'Very weak or no spontaneous movement',
    severity: 'critical',
  },
  {
    id: 'not_feeding',
    label: 'Unable to Feed',
    description: 'Poor sucking or refusing to breastfeed',
    severity: 'high',
  },
  {
    id: 'vomiting',
    label: 'Vomiting Everything',
    description: 'Cannot keep any feeds down',
    severity: 'high',
  },
  {
    id: 'temperature',
    label: 'Fever or Hypothermia',
    description: 'Temperature above 37.5C or below 35.5C',
    severity: 'high',
  },
  {
    id: 'umbilicus',
    label: 'Red/Draining Umbilicus',
    description: 'Signs of umbilical infection',
    severity: 'medium',
  },
  {
    id: 'skin_pustules',
    label: 'Skin Pustules',
    description: 'Multiple pustules or boils',
    severity: 'medium',
  },
  {
    id: 'yellow_palms',
    label: 'Yellow Palms/Soles',
    description: 'Severe jaundice extending to palms/soles',
    severity: 'high',
  },
];

type Step = 'info' | 'symptoms' | 'jaundice' | 'cry' | 'analysis';

export default function NewbornScreen({
  navigation,
}: NewbornScreenProps): React.JSX.Element {
  const [currentStep, setCurrentStep] = useState<Step>('info');
  const [newbornInfo, setNewbornInfo] = useState<NewbornInfo>({
    ageHours: '',
    birthWeight: '',
    deliveryType: '',
    apgarScore: '',
  });
  const [selectedSigns, setSelectedSigns] = useState<string[]>([]);
  const [skinImage, setSkinImage] = useState<string | null>(null);
  const [cryAudio, setCryAudio] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [_isAnalyzing, setIsAnalyzing] = useState(false);
  const [useEdgeAI] = useState(false);

  const recordingRef = useRef<Audio.Recording | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const toggleDangerSign = useCallback((signId: string) => {
    setSelectedSigns((prev) =>
      prev.includes(signId)
        ? prev.filter((id) => id !== signId)
        : [...prev, signId]
    );
  }, []);

  const captureSkin = async (): Promise<void> => {
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
      setSkinImage(result.assets[0].uri);
    }
  };

  const startRecording = async (): Promise<void> => {
    try {
      await Audio.requestPermissionsAsync();
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      recordingRef.current = recording;
      setIsRecording(true);
      setRecordingDuration(0);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingDuration((prev) => {
          if (prev >= 10) {
            stopRecording();
            return prev;
          }
          return prev + 1;
        });
      }, 1000);
    } catch (error) {
      Alert.alert('Error', 'Failed to start recording');
      console.error('Recording error:', error);
    }
  };

  const stopRecording = async (): Promise<void> => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (recordingRef.current) {
      try {
        await recordingRef.current.stopAndUnloadAsync();
        const uri = recordingRef.current.getURI();
        setCryAudio(uri);
        recordingRef.current = null;
      } catch (error) {
        console.error('Stop recording error:', error);
      }
    }
    setIsRecording(false);
  };

  const runAssessment = async (): Promise<void> => {
    setIsAnalyzing(true);
    setCurrentStep('analysis');

    try {
      // Run jaundice analysis
      let jaundiceResult = null;
      if (skinImage) {
        if (useEdgeAI && edgeAI.isEdgeAvailable()) {
          jaundiceResult = await edgeAI.analyzeJaundice(skinImage);
        } else {
          jaundiceResult = await analyzeJaundice(skinImage);
        }
      }

      // Run cry analysis
      let cryResult = null;
      if (cryAudio) {
        if (useEdgeAI && edgeAI.isEdgeAvailable()) {
          cryResult = await edgeAI.analyzeCry(cryAudio);
        } else {
          cryResult = await analyzeCry(cryAudio);
        }
      }

      // Build symptom summary
      const symptomSummary = selectedSigns
        .map((id) => DANGER_SIGNS.find((s) => s.id === id)?.label)
        .filter(Boolean)
        .join(', ');

      // Run combined assessment with MedGemma
      const combinedResult = await runCombinedAssessment({
        conjunctivaImage: null,
        skinImage,
        cryAudio,
      });

      // Compile results
      const results = {
        patient_type: 'newborn',
        age_hours: newbornInfo.ageHours,
        birth_weight: newbornInfo.birthWeight,
        delivery_type: newbornInfo.deliveryType,
        apgar_score: newbornInfo.apgarScore,
        danger_signs: selectedSigns,
        danger_signs_count: selectedSigns.length,
        has_critical_signs: selectedSigns.some(
          (id) => DANGER_SIGNS.find((s) => s.id === id)?.severity === 'critical'
        ),
        jaundice: jaundiceResult,
        cry_analysis: cryResult,
        symptoms_reported: symptomSummary,
        ...combinedResult,
        severity_level: determineSeverity(selectedSigns, jaundiceResult as Record<string, unknown> | null, cryResult as Record<string, unknown> | null),
      };

      navigation.navigate('Results', { type: 'combined', results });
    } catch (error) {
      Alert.alert('Error', 'Failed to complete assessment. Please try again.');
      console.error('Assessment error:', error);
      setCurrentStep('cry');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const determineSeverity = (
    signs: string[],
    jaundice: Record<string, unknown> | null,
    cry: Record<string, unknown> | null
  ): string => {
    const hasCriticalSigns = signs.some(
      (id) => DANGER_SIGNS.find((s) => s.id === id)?.severity === 'critical'
    );
    const hasHighSeveritySigns = signs.some(
      (id) => DANGER_SIGNS.find((s) => s.id === id)?.severity === 'high'
    );

    if (hasCriticalSigns) return 'RED';
    if (hasHighSeveritySigns) return 'RED';
    if (jaundice && (jaundice as { needs_phototherapy?: boolean }).needs_phototherapy) return 'RED';
    if (cry && (cry as { is_abnormal?: boolean }).is_abnormal) return 'YELLOW';
    if (signs.length > 1) return 'YELLOW';
    return 'GREEN';
  };

  const canProceed = (): boolean => {
    switch (currentStep) {
      case 'info':
        return newbornInfo.ageHours.length > 0;
      case 'symptoms':
        return true;
      case 'jaundice':
        return true;
      case 'cry':
        return true;
      default:
        return false;
    }
  };

  const nextStep = (): void => {
    const steps: Step[] = ['info', 'symptoms', 'jaundice', 'cry', 'analysis'];
    const currentIndex = steps.indexOf(currentStep);
    if (currentIndex < steps.length - 1) {
      setCurrentStep(steps[currentIndex + 1]);
    }
  };

  const prevStep = (): void => {
    const steps: Step[] = ['info', 'symptoms', 'jaundice', 'cry', 'analysis'];
    const currentIndex = steps.indexOf(currentStep);
    if (currentIndex > 0) {
      setCurrentStep(steps[currentIndex - 1]);
    }
  };

  const getStepNumber = (): number => {
    const steps: Step[] = ['info', 'symptoms', 'jaundice', 'cry', 'analysis'];
    return steps.indexOf(currentStep) + 1;
  };

  const renderNewbornInfo = (): React.JSX.Element => (
    <View style={styles.stepContent}>
      <Text style={styles.stepTitle}>Newborn Information</Text>
      <Text style={styles.stepDescription}>Enter basic newborn information</Text>

      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>Age (hours since birth)</Text>
        <TextInput
          style={styles.input}
          placeholder="e.g., 24"
          keyboardType="numeric"
          value={newbornInfo.ageHours}
          onChangeText={(text) =>
            setNewbornInfo((prev) => ({ ...prev, ageHours: text }))
          }
        />
      </View>

      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>Birth Weight (grams)</Text>
        <TextInput
          style={styles.input}
          placeholder="e.g., 3200"
          keyboardType="numeric"
          value={newbornInfo.birthWeight}
          onChangeText={(text) =>
            setNewbornInfo((prev) => ({ ...prev, birthWeight: text }))
          }
        />
      </View>

      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>Delivery Type</Text>
        <View style={styles.deliveryOptions}>
          <TouchableOpacity
            style={[
              styles.deliveryOption,
              newbornInfo.deliveryType === 'vaginal' && styles.deliveryOptionSelected,
            ]}
            onPress={() =>
              setNewbornInfo((prev) => ({ ...prev, deliveryType: 'vaginal' }))
            }
          >
            <Text
              style={[
                styles.deliveryOptionText,
                newbornInfo.deliveryType === 'vaginal' &&
                  styles.deliveryOptionTextSelected,
              ]}
            >
              Vaginal
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.deliveryOption,
              newbornInfo.deliveryType === 'cesarean' && styles.deliveryOptionSelected,
            ]}
            onPress={() =>
              setNewbornInfo((prev) => ({ ...prev, deliveryType: 'cesarean' }))
            }
          >
            <Text
              style={[
                styles.deliveryOptionText,
                newbornInfo.deliveryType === 'cesarean' &&
                  styles.deliveryOptionTextSelected,
              ]}
            >
              Cesarean
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>APGAR Score (if known)</Text>
        <TextInput
          style={styles.input}
          placeholder="e.g., 8"
          keyboardType="numeric"
          value={newbornInfo.apgarScore}
          onChangeText={(text) =>
            setNewbornInfo((prev) => ({ ...prev, apgarScore: text }))
          }
        />
      </View>
    </View>
  );

  const renderSymptoms = (): React.JSX.Element => (
    <View style={styles.stepContent}>
      <Text style={styles.stepTitle}>Danger Signs Checklist</Text>
      <Text style={styles.stepDescription}>
        Check any symptoms present (WHO IMNCI Newborn)
      </Text>

      <View style={styles.symptomsContainer}>
        {DANGER_SIGNS.map((sign) => (
          <TouchableOpacity
            key={sign.id}
            style={[
              styles.symptomCard,
              selectedSigns.includes(sign.id) && styles.symptomCardSelected,
              sign.severity === 'critical' && styles.symptomCardCritical,
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
            {sign.severity === 'critical' && (
              <View style={styles.criticalBadge}>
                <Text style={styles.criticalBadgeText}>CRITICAL</Text>
              </View>
            )}
          </TouchableOpacity>
        ))}
      </View>

      {selectedSigns.length > 0 && (
        <View
          style={[
            styles.selectedCount,
            selectedSigns.some(
              (id) => DANGER_SIGNS.find((s) => s.id === id)?.severity === 'critical'
            ) && styles.selectedCountCritical,
          ]}
        >
          <Ionicons
            name="warning"
            size={20}
            color={
              selectedSigns.some(
                (id) => DANGER_SIGNS.find((s) => s.id === id)?.severity === 'critical'
              )
                ? '#721c24'
                : '#856404'
            }
          />
          <Text
            style={[
              styles.selectedCountText,
              selectedSigns.some(
                (id) => DANGER_SIGNS.find((s) => s.id === id)?.severity === 'critical'
              ) && styles.selectedCountTextCritical,
            ]}
          >
            {selectedSigns.length} danger sign(s) selected
          </Text>
        </View>
      )}
    </View>
  );

  const renderJaundiceCapture = (): React.JSX.Element => (
    <View style={styles.stepContent}>
      <Text style={styles.stepTitle}>Jaundice Screening</Text>
      <Text style={styles.stepDescription}>
        Capture a photo of the newborn&apos;s skin (forehead or chest)
      </Text>

      <View style={styles.captureContainer}>
        {skinImage ? (
          <Image source={{ uri: skinImage }} style={styles.capturedImage} />
        ) : (
          <View style={styles.capturePlaceholder}>
            <Ionicons name="sunny-outline" size={64} color="#f39c12" />
            <Text style={styles.capturePlaceholderText}>No image captured</Text>
          </View>
        )}
      </View>

      <TouchableOpacity style={styles.captureButton} onPress={captureSkin}>
        <Ionicons name="camera" size={24} color="#fff" />
        <Text style={styles.captureButtonText}>
          {skinImage ? 'Retake Photo' : 'Take Photo'}
        </Text>
      </TouchableOpacity>

      <View style={styles.captureInstructions}>
        <Text style={styles.instructionsTitle}>Kramer Zone Assessment:</Text>
        <View style={styles.instructionItem}>
          <Text style={styles.kramerZone}>Zone 1:</Text>
          <Text style={styles.instructionText}>Face only (5-6 mg/dL)</Text>
        </View>
        <View style={styles.instructionItem}>
          <Text style={styles.kramerZone}>Zone 2:</Text>
          <Text style={styles.instructionText}>Chest/upper abdomen (8-10 mg/dL)</Text>
        </View>
        <View style={styles.instructionItem}>
          <Text style={styles.kramerZone}>Zone 3:</Text>
          <Text style={styles.instructionText}>Lower abdomen/thighs (11-13 mg/dL)</Text>
        </View>
        <View style={styles.instructionItem}>
          <Text style={styles.kramerZone}>Zone 4:</Text>
          <Text style={styles.instructionText}>Arms/lower legs (13-15 mg/dL)</Text>
        </View>
        <View style={styles.instructionItem}>
          <Text style={styles.kramerZone}>Zone 5:</Text>
          <Text style={styles.instructionText}>Hands/feet (&gt;15 mg/dL) - URGENT</Text>
        </View>
      </View>
    </View>
  );

  const renderCryCapture = (): React.JSX.Element => (
    <View style={styles.stepContent}>
      <Text style={styles.stepTitle}>Cry Analysis</Text>
      <Text style={styles.stepDescription}>
        Record 5-10 seconds of the infant&apos;s cry for asphyxia screening
      </Text>

      <View style={styles.recordingContainer}>
        <View
          style={[
            styles.recordingIndicator,
            isRecording && styles.recordingIndicatorActive,
            cryAudio && !isRecording && styles.recordingIndicatorComplete,
          ]}
        >
          <Ionicons
            name={isRecording ? 'radio-button-on' : cryAudio ? 'checkmark' : 'mic'}
            size={64}
            color={isRecording ? '#e74c3c' : cryAudio ? '#28a745' : '#9b59b6'}
          />
        </View>

        <Text style={styles.recordingStatus}>
          {isRecording
            ? `Recording: ${recordingDuration}s`
            : cryAudio
            ? 'Recording complete'
            : 'Ready to record'}
        </Text>

        {isRecording && (
          <View style={styles.recordingProgress}>
            <View
              style={[
                styles.recordingProgressFill,
                { width: `${(recordingDuration / 10) * 100}%` },
              ]}
            />
          </View>
        )}
      </View>

      <TouchableOpacity
        style={[
          styles.recordButton,
          isRecording && styles.recordButtonActive,
        ]}
        onPress={isRecording ? stopRecording : startRecording}
      >
        <Ionicons
          name={isRecording ? 'stop' : 'mic'}
          size={24}
          color="#fff"
        />
        <Text style={styles.recordButtonText}>
          {isRecording ? 'Stop Recording' : cryAudio ? 'Record Again' : 'Start Recording'}
        </Text>
      </TouchableOpacity>

      <View style={styles.cryInstructions}>
        <Text style={styles.instructionsTitle}>Tips for good recording:</Text>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={16} color="#28a745" />
          <Text style={styles.instructionText}>
            Hold phone 15-30 cm from baby
          </Text>
        </View>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={16} color="#28a745" />
          <Text style={styles.instructionText}>
            Minimize background noise
          </Text>
        </View>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={16} color="#28a745" />
          <Text style={styles.instructionText}>
            Record 5-10 seconds of crying
          </Text>
        </View>
      </View>

      <View style={styles.modelInfo}>
        <Ionicons name="analytics" size={20} color="#9b59b6" />
        <Text style={styles.modelInfoText}>
          Powered by HeAR (Health Acoustic Representations)
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
            <Text style={styles.analysisStepText}>Newborn info collected</Text>
          </View>
          <View style={styles.analysisStep}>
            <Ionicons name="checkmark-circle" size={20} color="#28a745" />
            <Text style={styles.analysisStepText}>
              {selectedSigns.length} danger signs evaluated
            </Text>
          </View>
          <View style={styles.analysisStep}>
            {skinImage ? (
              <Ionicons name="hourglass" size={20} color="#1a73e8" />
            ) : (
              <Ionicons name="remove-circle" size={20} color="#999" />
            )}
            <Text style={styles.analysisStepText}>
              {skinImage ? 'Jaundice screening with MedSigLIP...' : 'Jaundice screening skipped'}
            </Text>
          </View>
          <View style={styles.analysisStep}>
            {cryAudio ? (
              <Ionicons name="hourglass" size={20} color="#1a73e8" />
            ) : (
              <Ionicons name="remove-circle" size={20} color="#999" />
            )}
            <Text style={styles.analysisStepText}>
              {cryAudio ? 'Cry analysis with HeAR...' : 'Cry analysis skipped'}
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
            style={[styles.progressFill, { width: `${(getStepNumber() / 5) * 100}%` }]}
          />
        </View>
        <Text style={styles.progressText}>Step {getStepNumber()} of 5</Text>
      </View>

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        {currentStep === 'info' && renderNewbornInfo()}
        {currentStep === 'symptoms' && renderSymptoms()}
        {currentStep === 'jaundice' && renderJaundiceCapture()}
        {currentStep === 'cry' && renderCryCapture()}
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
              currentStep === 'cry' && styles.submitButton,
              !canProceed() && styles.nextButtonDisabled,
            ]}
            onPress={currentStep === 'cry' ? runAssessment : nextStep}
            disabled={!canProceed()}
          >
            <Text style={styles.nextButtonText}>
              {currentStep === 'cry' ? 'Run Assessment' : 'Next'}
            </Text>
            <Ionicons
              name={currentStep === 'cry' ? 'analytics' : 'arrow-forward'}
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
  deliveryOptions: {
    flexDirection: 'row',
    gap: 12,
  },
  deliveryOption: {
    flex: 1,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  deliveryOptionSelected: {
    borderColor: '#1a73e8',
    backgroundColor: '#e8f4fd',
  },
  deliveryOptionText: {
    fontSize: 16,
    color: '#666',
  },
  deliveryOptionTextSelected: {
    color: '#1a73e8',
    fontWeight: '600',
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
  symptomCardCritical: {
    borderLeftWidth: 4,
    borderLeftColor: '#dc3545',
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
  criticalBadge: {
    backgroundColor: '#f8d7da',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  criticalBadgeText: {
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
  selectedCountCritical: {
    backgroundColor: '#f8d7da',
  },
  selectedCountText: {
    marginLeft: 8,
    fontSize: 14,
    color: '#856404',
    fontWeight: '600',
  },
  selectedCountTextCritical: {
    color: '#721c24',
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
    backgroundColor: '#fff9e6',
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
    backgroundColor: '#f39c12',
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
    backgroundColor: '#fff9e6',
    padding: 16,
    borderRadius: 12,
  },
  instructionsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#856404',
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
  kramerZone: {
    fontSize: 13,
    fontWeight: '600',
    color: '#856404',
    width: 60,
  },
  recordingContainer: {
    alignItems: 'center',
    paddingVertical: 32,
  },
  recordingIndicator: {
    width: 150,
    height: 150,
    borderRadius: 75,
    backgroundColor: '#f0f0f0',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  recordingIndicatorActive: {
    backgroundColor: '#ffebee',
  },
  recordingIndicatorComplete: {
    backgroundColor: '#e8f5e9',
  },
  recordingStatus: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 16,
  },
  recordingProgress: {
    width: '80%',
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  recordingProgressFill: {
    height: '100%',
    backgroundColor: '#e74c3c',
    borderRadius: 4,
  },
  recordButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#9b59b6',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 12,
    marginBottom: 24,
  },
  recordButtonActive: {
    backgroundColor: '#e74c3c',
  },
  recordButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 8,
  },
  cryInstructions: {
    backgroundColor: '#f3e5f5',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
  },
  modelInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
  },
  modelInfoText: {
    marginLeft: 8,
    fontSize: 12,
    color: '#9b59b6',
    fontWeight: '600',
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
