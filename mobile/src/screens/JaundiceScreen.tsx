/**
 * Jaundice Detection Screen
 *
 * Captures neonatal skin/sclera image for jaundice detection
 * Uses MedSigLIP for zero-shot classification
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  ScrollView,
  Alert,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import type { RootStackParamList } from '../../App';
import { analyzeJaundice } from '../services/nexusApi';

type JaundiceScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Jaundice'>;
};

export default function JaundiceScreen({ navigation }: JaundiceScreenProps): React.JSX.Element {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

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
      const results = await analyzeJaundice(imageUri);
      navigation.navigate('Results', { type: 'jaundice', results: results as unknown as Record<string, unknown> });
    } catch (error) {
      Alert.alert('Analysis Error', 'Failed to analyze image. Please try again.');
      console.error('Jaundice analysis error:', error);
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
            Undress baby to expose skin (chest, abdomen, or thighs)
          </Text>
        </View>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={20} color="#28a745" />
          <Text style={styles.instructionText}>
            Use natural daylight (avoid yellow artificial light)
          </Text>
        </View>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={20} color="#28a745" />
          <Text style={styles.instructionText}>
            Press skin gently to blanch, then release and capture
          </Text>
        </View>
        <View style={styles.instructionItem}>
          <Ionicons name="checkmark-circle" size={20} color="#28a745" />
          <Text style={styles.instructionText}>
            For best results, capture the sclera (whites of eyes) too
          </Text>
        </View>
      </View>

      {/* Image Preview */}
      <View style={styles.imageContainer}>
        {imageUri ? (
          <Image source={{ uri: imageUri }} style={styles.imagePreview} />
        ) : (
          <View style={styles.placeholderContainer}>
            <Ionicons name="sunny-outline" size={64} color="#f39c12" />
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

      {/* Kramer Zones Info */}
      <View style={styles.kramerCard}>
        <Text style={styles.kramerTitle}>Kramer Zones Reference</Text>
        <View style={styles.kramerZone}>
          <View style={[styles.zoneIndicator, { backgroundColor: '#fff9c4' }]} />
          <Text style={styles.zoneText}>Zone 1 (Face): ~5-6 mg/dL</Text>
        </View>
        <View style={styles.kramerZone}>
          <View style={[styles.zoneIndicator, { backgroundColor: '#ffeb3b' }]} />
          <Text style={styles.zoneText}>Zone 2 (Chest): ~9 mg/dL</Text>
        </View>
        <View style={styles.kramerZone}>
          <View style={[styles.zoneIndicator, { backgroundColor: '#ffc107' }]} />
          <Text style={styles.zoneText}>Zone 3 (Abdomen): ~12 mg/dL</Text>
        </View>
        <View style={styles.kramerZone}>
          <View style={[styles.zoneIndicator, { backgroundColor: '#ff9800' }]} />
          <Text style={styles.zoneText}>Zone 4 (Arms/Legs): ~15 mg/dL</Text>
        </View>
        <View style={styles.kramerZone}>
          <View style={[styles.zoneIndicator, { backgroundColor: '#ff5722' }]} />
          <Text style={styles.zoneText}>Zone 5 (Hands/Feet): &gt;20 mg/dL</Text>
        </View>
      </View>

      {/* HAI-DEF Badge */}
      <View style={styles.modelInfo}>
        <Text style={styles.modelInfoText}>
          Powered by MedSigLIP (Google HAI-DEF)
        </Text>
        <Text style={styles.modelInfoSubtext}>
          Zero-shot jaundice detection with bilirubin estimation
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
    backgroundColor: '#fff8e1',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  instructionsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#f57c00',
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
    backgroundColor: '#f39c12',
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
  kramerCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  kramerTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  kramerZone: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  zoneIndicator: {
    width: 24,
    height: 24,
    borderRadius: 4,
    marginRight: 12,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  zoneText: {
    fontSize: 13,
    color: '#666',
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
