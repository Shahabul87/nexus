/**
 * Combined Assessment Screen
 *
 * Comprehensive assessment using all HAI-DEF models
 * Uses MedGemma for clinical synthesis
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import * as ImagePicker from 'expo-image-picker';
import { Audio } from 'expo-av';
import { Ionicons } from '@expo/vector-icons';
import type { RootStackParamList } from '../../App';
import { runCombinedAssessment } from '../services/nexusApi';

type CombinedAssessmentScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'CombinedAssessment'>;
};

interface AssessmentData {
  conjunctivaImage: string | null;
  skinImage: string | null;
  cryAudio: string | null;
}

export default function CombinedAssessmentScreen({
  navigation,
}: CombinedAssessmentScreenProps): React.JSX.Element {
  const [assessmentData, setAssessmentData] = useState<AssessmentData>({
    conjunctivaImage: null,
    skinImage: null,
    cryAudio: null,
  });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const recordingRef = React.useRef<Audio.Recording | null>(null);

  const captureImage = async (type: 'conjunctiva' | 'skin'): Promise<void> => {
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
      setAssessmentData((prev) => ({
        ...prev,
        [type === 'conjunctiva' ? 'conjunctivaImage' : 'skinImage']: result.assets[0].uri,
      }));
    }
  };

  const toggleRecording = async (): Promise<void> => {
    if (isRecording) {
      // Stop recording
      if (recordingRef.current) {
        await recordingRef.current.stopAndUnloadAsync();
        const uri = recordingRef.current.getURI();
        setAssessmentData((prev) => ({ ...prev, cryAudio: uri }));
        recordingRef.current = null;
      }
      setIsRecording(false);
    } else {
      // Start recording
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
      } catch (error) {
        Alert.alert('Error', 'Failed to start recording');
      }
    }
  };

  const runAssessment = async (): Promise<void> => {
    const hasData = Object.values(assessmentData).some((v) => v !== null);
    if (!hasData) {
      Alert.alert('No Data', 'Please capture at least one input.');
      return;
    }

    setIsAnalyzing(true);
    try {
      const results = await runCombinedAssessment(assessmentData);
      navigation.navigate('Results', { type: 'combined', results: results as unknown as Record<string, unknown> });
    } catch (error) {
      Alert.alert('Error', 'Failed to run assessment. Please try again.');
      console.error('Combined assessment error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getCompletionStatus = (): number => {
    let count = 0;
    if (assessmentData.conjunctivaImage) count++;
    if (assessmentData.skinImage) count++;
    if (assessmentData.cryAudio) count++;
    return count;
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Full Assessment</Text>
        <Text style={styles.headerSubtitle}>
          Capture multiple inputs for comprehensive analysis
        </Text>
        <View style={styles.progressContainer}>
          <Text style={styles.progressText}>
            {getCompletionStatus()}/3 inputs captured
          </Text>
          <View style={styles.progressBar}>
            <View
              style={[
                styles.progressFill,
                { width: `${(getCompletionStatus() / 3) * 100}%` },
              ]}
            />
          </View>
        </View>
      </View>

      {/* Input Cards */}
      <View style={styles.inputCardsContainer}>
        {/* Conjunctiva Card */}
        <TouchableOpacity
          style={[
            styles.inputCard,
            assessmentData.conjunctivaImage && styles.inputCardComplete,
          ]}
          onPress={() => captureImage('conjunctiva')}
        >
          {assessmentData.conjunctivaImage ? (
            <Image
              source={{ uri: assessmentData.conjunctivaImage }}
              style={styles.inputImage}
            />
          ) : (
            <View style={styles.inputPlaceholder}>
              <Ionicons name="eye-outline" size={40} color="#e74c3c" />
            </View>
          )}
          <View style={styles.inputInfo}>
            <Text style={styles.inputTitle}>Conjunctiva</Text>
            <Text style={styles.inputSubtitle}>For anemia screening</Text>
            <View style={styles.inputBadge}>
              <Text style={styles.inputBadgeText}>MedSigLIP</Text>
            </View>
          </View>
          {assessmentData.conjunctivaImage ? (
            <Ionicons name="checkmark-circle" size={24} color="#28a745" />
          ) : (
            <Ionicons name="camera" size={24} color="#999" />
          )}
        </TouchableOpacity>

        {/* Skin Card */}
        <TouchableOpacity
          style={[
            styles.inputCard,
            assessmentData.skinImage && styles.inputCardComplete,
          ]}
          onPress={() => captureImage('skin')}
        >
          {assessmentData.skinImage ? (
            <Image
              source={{ uri: assessmentData.skinImage }}
              style={styles.inputImage}
            />
          ) : (
            <View style={styles.inputPlaceholder}>
              <Ionicons name="sunny-outline" size={40} color="#f39c12" />
            </View>
          )}
          <View style={styles.inputInfo}>
            <Text style={styles.inputTitle}>Skin/Sclera</Text>
            <Text style={styles.inputSubtitle}>For jaundice detection</Text>
            <View style={styles.inputBadge}>
              <Text style={styles.inputBadgeText}>MedSigLIP</Text>
            </View>
          </View>
          {assessmentData.skinImage ? (
            <Ionicons name="checkmark-circle" size={24} color="#28a745" />
          ) : (
            <Ionicons name="camera" size={24} color="#999" />
          )}
        </TouchableOpacity>

        {/* Cry Audio Card */}
        <TouchableOpacity
          style={[
            styles.inputCard,
            assessmentData.cryAudio && styles.inputCardComplete,
            isRecording && styles.inputCardRecording,
          ]}
          onPress={toggleRecording}
        >
          <View
            style={[
              styles.inputPlaceholder,
              isRecording && styles.inputPlaceholderRecording,
            ]}
          >
            <Ionicons
              name={isRecording ? 'stop' : assessmentData.cryAudio ? 'checkmark' : 'mic'}
              size={40}
              color={isRecording ? '#e74c3c' : '#9b59b6'}
            />
          </View>
          <View style={styles.inputInfo}>
            <Text style={styles.inputTitle}>
              {isRecording ? 'Recording...' : 'Infant Cry'}
            </Text>
            <Text style={styles.inputSubtitle}>For asphyxia detection</Text>
            <View style={styles.inputBadge}>
              <Text style={styles.inputBadgeText}>HeAR</Text>
            </View>
          </View>
          {assessmentData.cryAudio ? (
            <Ionicons name="checkmark-circle" size={24} color="#28a745" />
          ) : (
            <Ionicons name={isRecording ? 'radio-button-on' : 'mic'} size={24} color="#999" />
          )}
        </TouchableOpacity>
      </View>

      {/* Analysis Button */}
      <TouchableOpacity
        style={[
          styles.analyzeButton,
          getCompletionStatus() === 0 && styles.analyzeButtonDisabled,
        ]}
        onPress={runAssessment}
        disabled={getCompletionStatus() === 0 || isAnalyzing}
      >
        {isAnalyzing ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <>
            <Ionicons name="analytics" size={24} color="#fff" />
            <Text style={styles.analyzeButtonText}>
              Run Clinical Synthesis
            </Text>
          </>
        )}
      </TouchableOpacity>

      {/* MedGemma Info */}
      <View style={styles.medgemmaInfo}>
        <Ionicons name="sparkles" size={24} color="#1a73e8" />
        <View style={styles.medgemmaContent}>
          <Text style={styles.medgemmaTitle}>MedGemma Clinical AI</Text>
          <Text style={styles.medgemmaText}>
            All findings will be synthesized using MedGemma to provide
            integrated clinical recommendations following WHO IMNCI protocols.
          </Text>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  content: {
    padding: 16,
  },
  header: {
    marginBottom: 24,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 16,
  },
  progressContainer: {
    marginTop: 8,
  },
  progressText: {
    fontSize: 12,
    color: '#666',
    marginBottom: 8,
  },
  progressBar: {
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#1a73e8',
    borderRadius: 4,
  },
  inputCardsContainer: {
    marginBottom: 24,
  },
  inputCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  inputCardComplete: {
    borderWidth: 2,
    borderColor: '#28a745',
  },
  inputCardRecording: {
    borderWidth: 2,
    borderColor: '#e74c3c',
  },
  inputImage: {
    width: 60,
    height: 60,
    borderRadius: 8,
  },
  inputPlaceholder: {
    width: 60,
    height: 60,
    borderRadius: 8,
    backgroundColor: '#f5f5f5',
    justifyContent: 'center',
    alignItems: 'center',
  },
  inputPlaceholderRecording: {
    backgroundColor: '#ffebee',
  },
  inputInfo: {
    flex: 1,
    marginLeft: 16,
  },
  inputTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  inputSubtitle: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  inputBadge: {
    alignSelf: 'flex-start',
    backgroundColor: '#e8f0fe',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
    marginTop: 4,
  },
  inputBadgeText: {
    fontSize: 10,
    color: '#1a73e8',
    fontWeight: '600',
  },
  analyzeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1a73e8',
    paddingVertical: 16,
    borderRadius: 12,
    marginBottom: 24,
  },
  analyzeButtonDisabled: {
    backgroundColor: '#ccc',
  },
  analyzeButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 8,
  },
  medgemmaInfo: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#e8f0fe',
    padding: 16,
    borderRadius: 12,
  },
  medgemmaContent: {
    flex: 1,
    marginLeft: 12,
  },
  medgemmaTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1a73e8',
    marginBottom: 4,
  },
  medgemmaText: {
    fontSize: 13,
    color: '#666',
    lineHeight: 18,
  },
});
