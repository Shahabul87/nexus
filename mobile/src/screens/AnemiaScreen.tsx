/**
 * Anemia Screening Screen
 *
 * Captures conjunctiva (inner eyelid) image for anemia detection
 * Uses MedSigLIP for zero-shot classification
 *
 * Supports:
 * - Edge AI (on-device TFLite inference)
 * - Cloud API (FastAPI backend)
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  ScrollView,
  Alert,
  Switch,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import type { RootStackParamList } from '../../App';
import { analyzeAnemia } from '../services/nexusApi';
import { edgeAI } from '../services/edgeAI';

type AnemiaScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Anemia'>;
};

export default function AnemiaScreen({ navigation }: AnemiaScreenProps): React.JSX.Element {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [useEdgeAI, setUseEdgeAI] = useState(false);
  const [edgeAvailable, setEdgeAvailable] = useState(false);

  const [isDownloadingModels, setIsDownloadingModels] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<{current: number; total: number} | null>(null);

  // Use ref to track if download prompt has been shown (persists across renders)
  const hasPromptedDownloadRef = useRef(false);

  // Download edge models - defined before useEffect to avoid hoisting issues
  const downloadEdgeModels = useCallback(async (): Promise<void> => {
    if (edgeAI.isDownloadInProgress()) {
      Alert.alert('Download in Progress', 'Model download is already in progress.');
      return;
    }

    setIsDownloadingModels(true);
    setDownloadProgress(null);

    try {
      // Download with progress tracking
      await edgeAI.downloadModels(undefined, (current, total) => {
        setDownloadProgress({ current, total });
      });

      // Re-initialize after download
      const available = await edgeAI.initialize();
      setEdgeAvailable(available);
      if (available) {
        setUseEdgeAI(true);
        Alert.alert(
          'Models Downloaded',
          'Edge AI models ready. Full on-device inference enabled for anemia, jaundice, and cry analysis. ' +
          'The app now works completely offline.'
        );
      } else {
        Alert.alert(
          'Download Complete',
          'Models downloaded but TensorFlow.js initialization failed. ' +
          'Will use cloud inference instead.'
        );
      }
    } catch (error) {
      console.error('Model download failed:', error);
      Alert.alert('Download Failed', 'Could not download edge models. Check your internet connection.');
    } finally {
      setIsDownloadingModels(false);
      setDownloadProgress(null);
    }
  }, []);

  // Initialize Edge AI on mount - prompt for download if not available
  useEffect(() => {
    const initEdge = async (): Promise<void> => {
      // First try without auto-download to check if models are already present
      const available = await edgeAI.initialize();
      setEdgeAvailable(available);
      if (available) {
        setUseEdgeAI(true);
      } else if (!hasPromptedDownloadRef.current) {
        // Prompt user to download models for offline capability
        hasPromptedDownloadRef.current = true;
        Alert.alert(
          'Enable Offline Mode?',
          'Download AI models (~27MB) for full offline functionality? ' +
          'This allows screenings without internet access.',
          [
            {
              text: 'Later',
              style: 'cancel',
            },
            {
              text: 'Download Now',
              onPress: downloadEdgeModels,
            },
          ]
        );
      }
    };
    initEdge();
  }, [downloadEdgeModels]);

  const pickImage = async (): Promise<void> => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permissionResult.granted) {
      Alert.alert('Permission Required', 'Please allow access to your photo library.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setImageUri(result.assets[0].uri);
    }
  };

  const takePhoto = async (): Promise<void> => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();

    if (!permissionResult.granted) {
      Alert.alert('Permission Required', 'Please allow access to your camera.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setImageUri(result.assets[0].uri);
    }
  };

  const analyzeImage = async (): Promise<void> => {
    if (!imageUri) {
      Alert.alert('No Image', 'Please capture or select an image first.');
      return;
    }

    setIsAnalyzing(true);
    try {
      let results;
      if (useEdgeAI && edgeAvailable) {
        // Use on-device inference
        results = await edgeAI.analyzeAnemia(imageUri);
      } else {
        // Use cloud API
        results = await analyzeAnemia(imageUri);
      }
      navigation.navigate('Results', { type: 'anemia', results: results as unknown as Record<string, unknown> });
    } catch (error) {
      Alert.alert('Analysis Error', 'Failed to analyze image. Please try again.');
      console.error('Anemia analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Instructions */}
      <View style={styles.instructionsCard}>
        <Text style={styles.instructionsTitle}>How to Capture</Text>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={20} color="#28a745" />
          <Text style={styles.instructionText}>
            Gently pull down the lower eyelid to expose the conjunctiva
          </Text>
        </View>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={20} color="#28a745" />
          <Text style={styles.instructionText}>
            Ensure good lighting (natural light preferred)
          </Text>
        </View>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={20} color="#28a745" />
          <Text style={styles.instructionText}>
            Hold camera steady and focus on the inner eyelid
          </Text>
        </View>
      </View>

      {/* Image Preview */}
      <View style={styles.imageContainer}>
        {imageUri ? (
          <Image source={{ uri: imageUri }} style={styles.imagePreview} />
        ) : (
          <View style={styles.placeholderContainer}>
            <Ionicons name="eye-outline" size={64} color="#ccc" />
            <Text style={styles.placeholderText}>No image selected</Text>
          </View>
        )}
      </View>

      {/* Capture Buttons */}
      <View style={styles.buttonRow}>
        <TouchableOpacity style={styles.captureButton} onPress={takePhoto}>
          <Ionicons name="camera" size={24} color="#fff" />
          <Text style={styles.buttonText}>Take Photo</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.captureButton, styles.galleryButton]}
          onPress={pickImage}
        >
          <Ionicons name="images" size={24} color="#1a73e8" />
          <Text style={[styles.buttonText, styles.galleryButtonText]}>Gallery</Text>
        </TouchableOpacity>
      </View>

      {/* Analyze Button */}
      <TouchableOpacity
        style={[
          styles.analyzeButton,
          !imageUri && styles.analyzeButtonDisabled,
        ]}
        onPress={analyzeImage}
        disabled={!imageUri || isAnalyzing}
      >
        {isAnalyzing ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <>
            <Ionicons name="scan" size={24} color="#fff" />
            <Text style={styles.analyzeButtonText}>Analyze with MedSigLIP</Text>
          </>
        )}
      </TouchableOpacity>

      {/* Edge AI Toggle */}
      <View style={styles.edgeToggle}>
        <View style={styles.edgeToggleInfo}>
          <Ionicons
            name={useEdgeAI ? 'phone-portrait' : 'cloud'}
            size={20}
            color={useEdgeAI ? '#28a745' : '#1a73e8'}
          />
          <Text style={styles.edgeToggleText}>
            {useEdgeAI ? 'On-Device (Offline)' : 'Cloud API'}
          </Text>
        </View>
        <Switch
          value={useEdgeAI}
          onValueChange={setUseEdgeAI}
          disabled={!edgeAvailable}
          trackColor={{ false: '#ccc', true: '#28a745' }}
          thumbColor={useEdgeAI ? '#fff' : '#f4f3f4'}
        />
      </View>
      {!edgeAvailable && (
        <View style={styles.edgeUnavailableContainer}>
          <Text style={styles.edgeUnavailable}>
            Edge AI models not downloaded. Using cloud inference.
          </Text>
          <TouchableOpacity
            style={styles.downloadButton}
            onPress={downloadEdgeModels}
            disabled={isDownloadingModels}
          >
            {isDownloadingModels ? (
              <View style={styles.downloadProgressContainer}>
                <ActivityIndicator color="#1a73e8" size="small" />
                {downloadProgress && (
                  <Text style={styles.downloadProgressText}>
                    {downloadProgress.current}/{downloadProgress.total} files
                  </Text>
                )}
              </View>
            ) : (
              <>
                <Ionicons name="cloud-download" size={18} color="#1a73e8" />
                <Text style={styles.downloadButtonText}>Download Models (~27MB)</Text>
              </>
            )}
          </TouchableOpacity>
          <Text style={styles.offlineNote}>
            Enables full offline operation for all screenings
          </Text>
        </View>
      )}

      {/* HAI-DEF Badge */}
      <View style={styles.modelInfo}>
        <Text style={styles.modelInfoText}>
          Powered by MedSigLIP (Google HAI-DEF)
        </Text>
        <Text style={styles.modelInfoSubtext}>
          {useEdgeAI && edgeAvailable
            ? 'Real on-device inference using TensorFlow.js'
            : 'Zero-shot anemia detection from conjunctiva imaging'}
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
    backgroundColor: '#e8f5e9',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  instructionsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2e7d32',
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
  imageContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 20,
    aspectRatio: 4 / 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  imagePreview: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  placeholderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f9f9f9',
  },
  placeholderText: {
    marginTop: 12,
    fontSize: 14,
    color: '#999',
  },
  buttonRow: {
    flexDirection: 'row',
    marginBottom: 20,
  },
  captureButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#e74c3c',
    paddingVertical: 14,
    borderRadius: 12,
    marginRight: 8,
  },
  galleryButton: {
    backgroundColor: '#fff',
    borderWidth: 2,
    borderColor: '#1a73e8',
    marginRight: 0,
    marginLeft: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  galleryButtonText: {
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
  edgeToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  edgeToggleInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  edgeToggleText: {
    marginLeft: 8,
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  edgeUnavailableContainer: {
    alignItems: 'center',
    marginBottom: 12,
  },
  edgeUnavailable: {
    fontSize: 12,
    color: '#999',
    textAlign: 'center',
    marginBottom: 8,
    fontStyle: 'italic',
  },
  downloadButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#e8f4fd',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#1a73e8',
  },
  downloadButtonText: {
    color: '#1a73e8',
    fontSize: 13,
    fontWeight: '600',
    marginLeft: 6,
  },
  downloadProgressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  downloadProgressText: {
    color: '#1a73e8',
    fontSize: 12,
    marginLeft: 8,
  },
  offlineNote: {
    fontSize: 11,
    color: '#666',
    marginTop: 6,
    fontStyle: 'italic',
    textAlign: 'center',
  },
});
