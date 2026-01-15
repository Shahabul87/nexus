/**
 * Cry Analysis Screen
 *
 * Records infant cry audio for asphyxia detection
 * Uses HeAR for health acoustic analysis
 */

import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  ScrollView,
  Alert,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Audio } from 'expo-av';
import { Ionicons } from '@expo/vector-icons';
import type { RootStackParamList } from '../../App';
import { analyzeCry } from '../services/nexusApi';

type CryAnalysisScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'CryAnalysis'>;
};

export default function CryAnalysisScreen({ navigation }: CryAnalysisScreenProps): React.JSX.Element {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [audioUri, setAudioUri] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  const recordingRef = useRef<Audio.Recording | null>(null);
  const soundRef = useRef<Audio.Sound | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

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
      setAudioUri(null);
      setRecordingDuration(0);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch (error) {
      Alert.alert('Recording Error', 'Failed to start recording.');
      console.error('Recording error:', error);
    }
  };

  const stopRecording = async (): Promise<void> => {
    if (!recordingRef.current) return;

    try {
      // Clear timer
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();

      setIsRecording(false);
      setAudioUri(uri);
      recordingRef.current = null;
    } catch (error) {
      Alert.alert('Recording Error', 'Failed to stop recording.');
      console.error('Stop recording error:', error);
    }
  };

  const playAudio = async (): Promise<void> => {
    if (!audioUri) return;

    try {
      if (soundRef.current) {
        await soundRef.current.unloadAsync();
      }

      const { sound } = await Audio.Sound.createAsync({ uri: audioUri });
      soundRef.current = sound;

      sound.setOnPlaybackStatusUpdate((status) => {
        if (status.isLoaded && status.didJustFinish) {
          setIsPlaying(false);
        }
      });

      setIsPlaying(true);
      await sound.playAsync();
    } catch (error) {
      Alert.alert('Playback Error', 'Failed to play audio.');
      console.error('Playback error:', error);
    }
  };

  const stopPlayback = async (): Promise<void> => {
    if (soundRef.current) {
      await soundRef.current.stopAsync();
      setIsPlaying(false);
    }
  };

  const analyzeAudio = async (): Promise<void> => {
    if (!audioUri) {
      Alert.alert('No Recording', 'Please record audio first.');
      return;
    }

    setIsAnalyzing(true);
    try {
      const results = await analyzeCry(audioUri);
      navigation.navigate('Results', { type: 'cry', results: results as unknown as Record<string, unknown> });
    } catch (error) {
      Alert.alert('Analysis Error', 'Failed to analyze audio. Please try again.');
      console.error('Cry analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Instructions */}
      <View style={styles.instructionsCard}>
        <Text style={styles.instructionsTitle}>How to Record</Text>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={20} color="#28a745" />
          <Text style={styles.instructionText}>
            Record when infant is actively crying (not sleeping)
          </Text>
        </View>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={20} color="#28a745" />
          <Text style={styles.instructionText}>
            Hold phone 20-30 cm from baby's mouth
          </Text>
        </View>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={20} color="#28a745" />
          <Text style={styles.instructionText}>
            Record at least 5 seconds of continuous crying
          </Text>
        </View>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={20} color="#28a745" />
          <Text style={styles.instructionText}>
            Minimize background noise if possible
          </Text>
        </View>
      </View>

      {/* Recording Interface */}
      <View style={styles.recordingContainer}>
        <View style={styles.waveformPlaceholder}>
          {isRecording ? (
            <View style={styles.recordingIndicator}>
              <Ionicons name="radio-button-on" size={24} color="#e74c3c" />
              <Text style={styles.recordingText}>Recording...</Text>
            </View>
          ) : audioUri ? (
            <View style={styles.recordingIndicator}>
              <Ionicons name="checkmark-circle" size={24} color="#28a745" />
              <Text style={styles.recordingText}>Recording saved</Text>
            </View>
          ) : (
            <View style={styles.recordingIndicator}>
              <Ionicons name="mic-outline" size={48} color="#ccc" />
              <Text style={styles.placeholderText}>Ready to record</Text>
            </View>
          )}
        </View>

        <Text style={styles.durationText}>
          {formatDuration(recordingDuration)}
        </Text>

        {/* Record Button */}
        <TouchableOpacity
          style={[
            styles.recordButton,
            isRecording && styles.recordButtonActive,
          ]}
          onPress={isRecording ? stopRecording : startRecording}
        >
          <Ionicons
            name={isRecording ? 'stop' : 'mic'}
            size={40}
            color="#fff"
          />
        </TouchableOpacity>

        <Text style={styles.recordHint}>
          {isRecording ? 'Tap to stop' : 'Tap to record'}
        </Text>
      </View>

      {/* Playback Controls */}
      {audioUri && (
        <View style={styles.playbackContainer}>
          <TouchableOpacity
            style={styles.playButton}
            onPress={isPlaying ? stopPlayback : playAudio}
          >
            <Ionicons
              name={isPlaying ? 'pause' : 'play'}
              size={24}
              color="#1a73e8"
            />
            <Text style={styles.playButtonText}>
              {isPlaying ? 'Pause' : 'Play Recording'}
            </Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Analyze Button */}
      <TouchableOpacity
        style={[
          styles.analyzeButton,
          !audioUri && styles.analyzeButtonDisabled,
        ]}
        onPress={analyzeAudio}
        disabled={!audioUri || isAnalyzing || isRecording}
      >
        {isAnalyzing ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <>
            <Ionicons name="analytics" size={24} color="#fff" />
            <Text style={styles.analyzeButtonText}>Analyze with HeAR</Text>
          </>
        )}
      </TouchableOpacity>

      {/* Info Cards */}
      <View style={styles.infoCard}>
        <Text style={styles.infoTitle}>What we detect</Text>
        <View style={styles.infoItem}>
          <Text style={styles.infoLabel}>Birth Asphyxia Signs:</Text>
          <Text style={styles.infoValue}>
            High-pitched cry, irregular patterns
          </Text>
        </View>
        <View style={styles.infoItem}>
          <Text style={styles.infoLabel}>Cry Types:</Text>
          <Text style={styles.infoValue}>
            Pain, hunger, discomfort, tiredness
          </Text>
        </View>
      </View>

      {/* HAI-DEF Badge */}
      <View style={styles.modelInfo}>
        <Text style={styles.modelInfoText}>
          Powered by HeAR (Google HAI-DEF)
        </Text>
        <Text style={styles.modelInfoSubtext}>
          Health Acoustic Representations for cry analysis
        </Text>
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
  instructionsCard: {
    backgroundColor: '#f3e5f5',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  instructionsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#7b1fa2',
    marginBottom: 12,
  },
  instructionItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  instructionText: {
    flex: 1,
    marginLeft: 8,
    fontSize: 14,
    color: '#333',
  },
  recordingContainer: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  waveformPlaceholder: {
    width: '100%',
    height: 100,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f9f9f9',
    borderRadius: 12,
    marginBottom: 16,
  },
  recordingIndicator: {
    alignItems: 'center',
  },
  recordingText: {
    marginTop: 8,
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  placeholderText: {
    marginTop: 8,
    fontSize: 14,
    color: '#999',
  },
  durationText: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 20,
    fontVariant: ['tabular-nums'],
  },
  recordButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#9b59b6',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
    shadowColor: '#9b59b6',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  recordButtonActive: {
    backgroundColor: '#e74c3c',
  },
  recordHint: {
    fontSize: 14,
    color: '#666',
  },
  playbackContainer: {
    marginBottom: 20,
  },
  playButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#1a73e8',
  },
  playButtonText: {
    marginLeft: 8,
    fontSize: 16,
    fontWeight: '600',
    color: '#1a73e8',
  },
  analyzeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1a73e8',
    paddingVertical: 16,
    borderRadius: 12,
    marginBottom: 20,
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
  infoCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  infoTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  infoItem: {
    marginBottom: 8,
  },
  infoLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#666',
  },
  infoValue: {
    fontSize: 13,
    color: '#999',
  },
  modelInfo: {
    alignItems: 'center',
    paddingVertical: 16,
  },
  modelInfoText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
  },
  modelInfoSubtext: {
    fontSize: 12,
    color: '#999',
    marginTop: 4,
  },
});
